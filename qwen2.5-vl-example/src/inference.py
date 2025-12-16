"""
Qwen2.5-VL 推理脚本
支持图像和视频输入，可选 PSA 稀疏注意力加速
"""
import argparse
import os
import sys
import json
import subprocess
import tempfile
import torch

# 添加 src 目录到路径，支持从项目根目录运行
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def get_video_info(video_path: str) -> dict:
    """获取视频信息"""
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate,duration,nb_frames',
        '-show_entries', 'format=duration',
        '-of', 'json',
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)
    stream = info['streams'][0]

    # 计算帧率
    fps = None
    if 'r_frame_rate' in stream:
        num, den = map(int, stream['r_frame_rate'].split('/'))
        fps = num / den

    # 获取时长
    duration = float(stream.get('duration', info['format'].get('duration', 0)))

    # 获取总帧数
    nb_frames = int(stream.get('nb_frames', 0))
    if nb_frames == 0 and fps and duration:
        nb_frames = int(fps * duration)

    return {'fps': fps, 'duration': duration, 'nb_frames': nb_frames}


def preprocess_video(input_path: str, target_frames: int = 64, quality: int = 23) -> str:
    """
    预处理视频到指定帧数

    Args:
        input_path: 输入视频路径
        target_frames: 目标帧数
        quality: CRF质量值

    Returns:
        预处理后的临时视频路径
    """
    info = get_video_info(input_path)
    print(f"原始视频: {info['nb_frames']} 帧, {info['duration']:.1f}秒")

    target_fps = target_frames / info['duration']
    print(f"转换到 {target_frames} 帧 (fps: {target_fps:.4f})")

    temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4', prefix='qwen_preprocess_')
    os.close(temp_fd)

    cmd = [
        'ffmpeg', '-i', input_path,
        '-r', str(target_fps),
        '-c:v', 'libx264',
        '-crf', str(quality),
        '-c:a', 'copy',
        '-y', temp_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        os.unlink(temp_path)
        raise RuntimeError(f"视频预处理失败: {result.stderr}")

    output_info = get_video_info(temp_path)
    print(f"预处理完成: {output_info['nb_frames']} 帧")
    return temp_path


class QwenVLInference:
    """Qwen2.5-VL 推理类"""

    # 默认缓存目录
    DEFAULT_CACHE_DIR = "/workspace/qwenvl2.5-clean/models"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "auto",
        dtype: str = "auto",
        use_psa: bool = False,
        psa_log_dir: str = None,
        cache_dir: str = None,
    ):
        """
        初始化模型

        Args:
            model_name: 模型名称或路径
            device: 设备 (auto, cuda, cpu)
            dtype: 数据类型 (auto, float16, bfloat16, float32)
            use_psa: 是否使用 PSA 稀疏注意力
            psa_log_dir: PSA 日志目录
            cache_dir: 模型缓存目录 (默认: /workspace/qwenvl2.5-clean/models)
        """
        self.model_name = model_name
        self.use_psa = use_psa
        self.cache_dir = cache_dir or self.DEFAULT_CACHE_DIR

        # 确保缓存目录存在
        os.makedirs(self.cache_dir, exist_ok=True)

        # 选择数据类型
        if dtype == "auto":
            self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        else:
            self.torch_dtype = getattr(torch, dtype)

        # 选择设备
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"加载模型: {model_name}")
        print(f"设备: {self.device}, 数据类型: {self.torch_dtype}")
        print(f"缓存目录: {self.cache_dir}")

        # 加载模型
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            device_map=device if device != "cpu" else None,
            attn_implementation="sdpa",
            cache_dir=self.cache_dir,
        )
        if device == "cpu":
            self.model = self.model.to(self.device)

        # 应用 PSA 稀疏注意力
        if use_psa:
            from attention import replace_psa_attention_qwen2vl, verify_attention_replacement
            self.model = replace_psa_attention_qwen2vl(self.model, log_dir=psa_log_dir)
            verify_attention_replacement(self.model)

        self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=self.cache_dir)
        print("模型加载完成")

    def inference_image(self, image_path: str, prompt: str, max_tokens: int = 512) -> str:
        """
        图像推理

        Args:
            image_path: 图像路径或URL
            prompt: 提示词
            max_tokens: 最大生成token数

        Returns:
            模型生成的文本
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self._generate(messages, max_tokens)

    def inference_video(
        self,
        video_path: str,
        prompt: str,
        max_tokens: int = 512,
        preprocess_frames: int = None,
    ) -> str:
        """
        视频推理

        Args:
            video_path: 视频路径
            prompt: 提示词
            max_tokens: 最大生成token数
            preprocess_frames: 预处理帧数 (可选，推荐长视频使用)

        Returns:
            模型生成的文本
        """
        temp_path = None
        try:
            if preprocess_frames:
                temp_path = preprocess_video(video_path, preprocess_frames)
                video_path = temp_path

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": video_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            return self._generate(messages, max_tokens)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    def inference_multi_image(self, image_paths: list, prompt: str, max_tokens: int = 512) -> str:
        """
        多图推理

        Args:
            image_paths: 图像路径列表
            prompt: 提示词
            max_tokens: 最大生成token数

        Returns:
            模型生成的文本
        """
        content = [{"type": "image", "image": path} for path in image_paths]
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]
        return self._generate(messages, max_tokens)

    def _generate(self, messages: list, max_tokens: int) -> str:
        """内部生成方法"""
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        return output_text[0]


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL 推理")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="模型名称或路径")
    parser.add_argument("--type", type=str, default="image", choices=["image", "video"],
                        help="输入类型: image 或 video")
    parser.add_argument("--input", type=str, required=True,
                        help="输入文件路径或URL")
    parser.add_argument("--prompt", type=str, default="请描述这张图片",
                        help="提示词")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="最大生成token数")
    parser.add_argument("--device", type=str, default="auto",
                        help="设备 (auto, cuda, cpu)")
    parser.add_argument("--dtype", type=str, default="auto",
                        choices=["auto", "float16", "bfloat16", "float32"],
                        help="数据类型")
    parser.add_argument("--preprocess-frames", type=int, default=None,
                        help="视频预处理帧数 (推荐长视频使用)")
    parser.add_argument("--use-psa", action="store_true",
                        help="使用 PSA 稀疏注意力加速")
    parser.add_argument("--psa-log-dir", type=str, default=None,
                        help="PSA 日志保存目录")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="模型缓存目录 (默认: /workspace/qwenvl2.5-clean/models)")

    args = parser.parse_args()

    # 初始化模型
    model = QwenVLInference(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        use_psa=args.use_psa,
        psa_log_dir=args.psa_log_dir,
        cache_dir=args.cache_dir,
    )

    # 推理
    print(f"\n输入: {args.input}")
    print(f"提示: {args.prompt}\n")

    if args.type == "image":
        result = model.inference_image(args.input, args.prompt, args.max_tokens)
    else:
        result = model.inference_video(
            args.input, args.prompt, args.max_tokens, args.preprocess_frames
        )

    print("=" * 50)
    print("模型回答:")
    print(result)
    print("=" * 50)


if __name__ == "__main__":
    main()
