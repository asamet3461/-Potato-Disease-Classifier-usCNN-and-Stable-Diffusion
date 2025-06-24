#  Potato Disease Classifier using CNN and Stable Diffusion  
#  CNN ve Stable Diffusion ile Patates Hastalık Sınıflandırması

##  Description | Açıklama

This project trains a convolutional neural network (CNN) to classify potato leaf diseases using labeled images.  
Additionally, it uses a Stable Diffusion pipeline to generate synthetic images of diseased potatoes based on text prompts.

Bu proje, etiketli görüntüler üzerinden patates yaprağı hastalıklarını sınıflandırmak için bir evrişimli sinir ağı (CNN) eğitir.  
Ayrıca, metin girdilerine göre yapay hasta patates görselleri üretmek için Stable Diffusion kullanır.

---

##  Features | Özellikler

- Custom image classification using PyTorch and torchvision  
- Multi-class support for several disease types  
- Data augmentation and transformations  
- Stable Diffusion for disease image generation

- PyTorch ve torchvision ile özel görüntü sınıflandırma  
- Birden fazla hastalık sınıfı desteği  
- Veri artırma ve dönüşümler  
- Stable Diffusion ile hastalıklı yaprak görsel üretimi

---

##  Output | Çıktı

- Trained CNN with accuracy metrics  
- Sample generated potato disease images using diffusion model  

- Eğitilmiş CNN doğruluk sonuçları  
- Stable Diffusion ile üretilmiş yapay hasta yaprak görselleri

---

##  Requirements | Gereksinimler

```bash
pip install torch torchvision diffusers matplotlib
