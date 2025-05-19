# Jet Tagging med CNN, ParticleNet og LorentzNet

## Oversikt
Dette repositoriet gir en komplett pipeline for trening og evaluering av tre forskjellige jet-tagging-modeller innen høyenergifysikk:
- **CNN (Convolutional Neural Network)**  
- **ParticleNet (Grafbasert modell)**  
- **LorentzNet (Lorentz-ekvariant modell)**  

Hver modell har sin egen mappe med skript for datakonvertering, trening og evaluering. Dette gir fleksibilitet og modularitet i eksperimenteringen med hver arkitektur.

---

## Installasjon

1. **Klone repositoriet:**  
   git clone https://github.com/DanielNSaga/JetTaggingComparison  
   

2. **Installer nødvendige pakker:**  
   pip install -r requirements.txt  

---

## Forberedelse av data

### 1. Last ned ROOT-filer  
Før du konverterer data for hver modell, må de nødvendige ROOT-filene lastes ned.

- Kjør følgende skript for å laste ned ROOT-filene:  
  python download_files.py  

- Dette vil laste ned de nødvendige dataene i en standardmappe.

---

## Data Konvertering og Trening

### 1. Velg modellen du ønsker å trene:  
- Gå inn i mappen for den aktuelle modellen:  
  - For CNN:  
    cd CNN  
  - For ParticleNet:  
    cd ParticleNet  
  - For LorentzNet:  
    cd LorentzNet  

### 2. Konverter ROOT-filene til HDF5:  
- Kjør convert_file.py i den valgte mappen for å konvertere dataene:  
  python convert_file.py  

- Dette vil generere trenings-, validerings- og testsett i HDF5-format, spesifikt tilpasset modellen.

### 3. Tren modellen:  
- Når konverteringen er fullført, kan du trene modellen med:  
  python trainer.py  

- Treningen logger progresjon, tap og nøyaktighet, og lagrer den beste modellen automatisk.

---

## Mappeoversikt
- **/CNN** - Inneholder kode og data for CNN-modellen.  
- **/ParticleNet** - Inneholder kode og data for ParticleNet-modellen.  
- **/LorentzNet** - Inneholder kode og data for LorentzNet-modellen.  
- **download_files.py** - Skript for å laste ned de nødvendige ROOT-filene.  
- **requirements.txt** - Liste over nødvendige Python-pakker.  

---

## Viktig: Kun for CUDA-kompatible GPUer
Dette repositoriet krever en NVIDIA GPU med CUDA-støtte for å kunne trene modellene effektivt. CPU-trening er ikke støttet, og andre GPU-typer (f.eks. AMD) støttes ikke uten omfattende modifikasjoner.

---


## Lisens
Dette repositoriet er lisensiert under MIT-lisensen. Se LICENSE-filen for detaljer.
