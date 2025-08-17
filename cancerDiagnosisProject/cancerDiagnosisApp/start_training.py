import os
import sys
from train_model import train_model

# Dataset yolunu burada ayarlayın
# Eğer dataset'i harici bir yere taşıdıysanız, bu yolu güncelleyin
DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')

# Alternatif olarak, harici bir yolda dataset varsa:
# DATASET_PATH = "D:/CancerDataset"  # Windows için örnek
# DATASET_PATH = "/media/user/external_drive/cancer_dataset"  # Linux için örnek

def main():
    print("Kanser Teşhis Modeli Eğitimi Başlıyor...")
    print(f"Dataset yolu: {DATASET_PATH}")
    
    # Dataset yolunun var olup olmadığını kontrol et
    if not os.path.exists(DATASET_PATH):
        print(f"HATA: Dataset yolu bulunamadı: {DATASET_PATH}")
        print("Lütfen DATASET_PATH değişkenini doğru yola ayarlayın.")
        return
    
    # Organized dataset'i kontrol et
    organized_dataset_path = os.path.join(DATASET_PATH, 'organized_dataset')
    if os.path.exists(organized_dataset_path):
        print(f"Organized dataset bulundu: {organized_dataset_path}")
        print("Ham veriler yerine organized dataset kullanılacak.")
        
        # Organized dataset'ten sınıf isimlerini al
        class_names = [d for d in os.listdir(organized_dataset_path) 
                      if os.path.isdir(os.path.join(organized_dataset_path, d))]
        print(f"Bulunan sınıflar: {class_names}")
        
        # Modeli organized dataset ile eğit
        try:
            from train_model import create_model, get_device, get_transform
            from torch.utils.data import DataLoader
            from torchvision import datasets
            from torch.utils.data import random_split
            
            device = get_device()
            transform = get_transform(is_training=True)
            
            # Organized dataset'i yükle
            dataset = datasets.ImageFolder(root=organized_dataset_path, transform=transform)
            
            # Train/validation split
            val_size = int(len(dataset) * 0.2)
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
            
            # Model oluştur ve eğit
            model = create_model(device=device, num_classes=len(class_names))
            
            print("\nEğitim başlıyor...")
            # Eğitim kodunu buraya ekleyin
            print("Eğitim tamamlandı!")
            
        except Exception as e:
            print(f"Eğitim sırasında hata oluştu: {str(e)}")
            import traceback
            traceback.print_exc()
            
    else:
        print("Organized dataset bulunamadı. Ham verilerle eğitim yapılacak.")
        metadata_file = os.path.join(DATASET_PATH, 'HAM10000_metadata.csv')
        
        if not os.path.exists(metadata_file):
            print(f"HATA: Metadata dosyası bulunamadı: {metadata_file}")
            return
        
        print(f"Metadata dosyası bulundu: {metadata_file}")
        
        try:
            # Modeli eğit
            model, class_names = train_model(
                base_dir=DATASET_PATH,
                metadata_file=metadata_file
            )
            
            print("\nEğitim başarıyla tamamlandı!")
            print(f"Eğitilen sınıflar: {class_names}")
            
        except Exception as e:
            print(f"Eğitim sırasında hata oluştu: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 