# feature_extractor.py
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np

#feature dim = 768

class DinoV2Finetuned(nn.Module):
    def __init__(self, model_name="facebook/dinov2-base", num_classes=768):
        super().__init__()
        self.base = AutoModel.from_pretrained(model_name)
        hidden = self.base.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden, 768),   
            nn.ReLU(),
            nn.Linear(768, num_classes)
        )

    def forward(self, pixel_values):
        out = self.base(pixel_values=pixel_values)
        pooled = out.pooler_output
        projected = self.projection(pooled)
        return projected


class FeatureExtractor:
    def __init__(self, model_name="facebook/dinov2-large", finetune_path=None, device='cuda'):
        """
        راه‌اندازی مدل استخراج ویژگی DINOv2
        model_name: یکی از مدل‌های DINOv2 (small, base, large, giant)
        finetune_path: مسیر فایل وزن‌های فاین‌تیون شده (اگر None باشد، فقط مدل پرترین در دسترس است)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = AutoImageProcessor.from_pretrained(model_name )

       
        self.model_pretrained = AutoModel.from_pretrained(model_name)
        self.model_pretrained.eval().to(self.device)



        if finetune_path is not None:
            self.model_finetuned = DinoV2Finetuned(model_name=model_name)
            state_dict = torch.load(finetune_path, map_location=self.device, weights_only=True)
            self.model_finetuned.load_state_dict(state_dict)
            self.model_finetuned.eval().to(self.device)
            print(f"[INFO] وزن‌های فاین‌تیون شده از {finetune_path} بارگذاری شد ✅")

      
        self.feature_dims = {}
        with torch.no_grad():
            dummy = torch.zeros((1, 3, 224, 224)).to(self.device)
           
            out_pretrained = self.model_pretrained(pixel_values=dummy).pooler_output
            self.feature_dims['pretrained'] = out_pretrained.shape[-1]
            # برای مدل فاین‌تیون اگر موجود باشد


        print(f"[INFO] مدل {model_name} بارگذاری شد. ابعاد ویژگی پرترین: {self.feature_dims.get('pretrained', 'ناموجود')}")
        if 'finetuned' in self.feature_dims:
            print(f"[INFO] ابعاد ویژگی فاین‌تیون: {self.feature_dims['finetuned']}")

    def extract_from_path(self, img_path: str, model_type='pretrained') -> np.ndarray:
        """
        گرفتن ویژگی‌های تصویر از مسیر محلی
        model_type: 'pretrained' برای مدل پرترین یا 'finetuned' برای مدل فاین‌تیون شده
        """
        img = Image.open(img_path).convert("RGB")
        return self._extract(img, model_type=model_type)

    def extract_image(self, image: Image.Image, model_type='pretrained') -> np.ndarray:
        """
        گرفتن ویژگی‌ها از یک شیء Image باز شده
        model_type: 'pretrained' برای مدل پرترین یا 'finetuned' برای مدل فاین‌تیون شده
        """
        return self._extract(image, model_type=model_type)

    def _extract(self, img: Image.Image, model_type='pretrained') -> np.ndarray:
        """
        پردازش و استخراج ویژگی از تصویر با DINOv2
        model_type: 'pretrained' یا 'finetuned'
        """
        if model_type not in ['pretrained', 'finetuned']:
            raise ValueError("model_type باید 'pretrained' یا 'finetuned' باشد")

        if model_type == 'finetuned' and self.model_finetuned is None:
            raise ValueError("مدل فاین‌تیون شده بارگذاری نشده است. finetune_path را در __init__ مشخص کنید.")

        
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            if model_type == 'pretrained':
                outputs = self.model_pretrained(**inputs)
                embeddings = outputs.pooler_output
            else:  # finetuned
                embeddings = self.model_finetuned(**inputs)

            embeddings = nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy().flatten()