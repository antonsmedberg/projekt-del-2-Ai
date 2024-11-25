
# **YOLOv5 med COCO-dataset för Objektigenkänning**

## **Projektbeskrivning**
Det här projektet implementerar en AI-pipeline för att träna och optimera en YOLOv5-modell med hjälp av COCO-datasetet. Målet är att skapa en robust och effektiv lösning för realtidsobjektigenkänning, där modellen kan identifiera och klassificera objekt i bilder med hög precision och snabbhet. 

Jag har designat projektet för att möta VG-nivåns krav, vilket inkluderar:
- Skriva välstrukturerad, väldokumenterad och lättläst kod.
- Välmotiverat val av algoritm och dataset.
- Implementera effektiv Pythonkod och robust felhantering.
- Utforska alternativ och förbättringsmöjligheter inom AI-metoder.

---

## **Motivering av val**
### **Val av algoritm: YOLOv5**
YOLOv5 är en välkänd och pålitlig algoritm för objektigenkänning, och jag valde den på grund av:
- **Effektivitet**: YOLOv5 balanserar noggrannhet och snabbhet, vilket gör den idealisk för realtidsapplikationer.
- **Flexibilitet**: Modellen är modulär och enkel att anpassa för olika dataset och applikationer.
- **Gemenskapsstöd**: Det finns omfattande dokumentation och en stor användarbas, vilket underlättar vidareutveckling.

### **Val av dataset: COCO**
COCO-datasetet (Common Objects in Context) är en industristandard för objektigenkänning med över 80 objektklasser. Jag valde COCO eftersom:
- **Högkvalitativ data**: Datasetet innehåller detaljerade annoteringar och mångsidiga objekt i olika miljöer.
- **Relevans**: Det används ofta för benchmarking av objektigenkänningsmodeller, vilket gör det möjligt att jämföra prestanda med andra modeller.

---

## **Projektmål**
Projektet syftar till att:
1. Utveckla en djup förståelse för YOLOv5-modellen och dess träningsprocess.
2. Implementera en komplett träningspipeline med datasetförberedelser, modellträning och validering.
3. Utforska och dokumentera optimeringsstrategier för att förbättra prestanda.
4. Exportera en tränad modell i ONNX-format för att möjliggöra bredare användning och kompatibilitet.

---

## **Projektstruktur**
Projektet är uppdelat i följande huvudkomponenter:

### **1. Datasetförberedelser**
- Automatisk nedladdning av COCO128, en mindre version av COCO-datasetet.
- Verifiering av datasetstruktur för att säkerställa att alla nödvändiga filer och mappar är korrekt organiserade.
- Generering av en `dataset.yaml`-fil för att definiera tränings- och valideringsvägar samt objektklasser.

### **2. Träningspipeline**
- **Parametrar**:
  - Bildstorlek: 640x640 pixlar.
  - Batchstorlek: 16.
  - Antal epoch: 50.
  - Optimerare: AdamW.
- Dynamisk validering av indata och träningsparametrar för att förhindra vanliga fel.
- Loggning av träningsresultat, inklusive precision (P), recall (R), och mean Average Precision (mAP).

### **3. Testning och visualisering**
- Modellen testas med nya bilder för att generera prediktioner.
- Annoterade bilder sparas för analys och utvärdering.

### **4. Export och distribution**
- Export av modellen till ONNX-format för att underlätta vidare användning på andra plattformar och i produktionsmiljöer.

---

## **Kodkvalitet och felhantering**
Projektet följer principerna för effektiv och lättläst Pythonkod:
- **Modularisering**: Koden är uppdelad i väldefinierade funktioner och moduler, vilket förbättrar underhållbarhet och läsbarhet.
- **Felhantering**: Varje kritisk del av pipeline, inklusive datasetvalidering, parameterinställningar och modellträning, har robust felhantering för att undvika avbrott.
- **Effektivitet**: Pipeline är optimerad för att utnyttja GPU-acceleration när det är möjligt.

### **Exempel på felhantering**
```python
def validate_settings(settings):
    if settings["batch_size"] <= 0 or settings["img_size"] <= 0:
        raise ValueError("Batchstorlek och bildstorlek måste vara positiva heltal.")
    logging.info("✅ Träningsparametrar validerade.")
```

---

## **Analys och förbättringsmöjligheter**
### **Vad jag har lärt mig**
- Hur YOLOv5:s arkitektur fungerar och kan anpassas för olika dataset.
- Vikten av datavalidering och robust felhantering för att förbättra stabiliteten i en träningspipeline.
- Hur man optimerar träningsparametrar för att balansera snabbhet och noggrannhet.

### **Förbättringsmöjligheter**
1. **Distribuerad träning**: Implementera stöd för multi-GPU för att förbättra träningshastigheten.
2. **Utökat dataset**: Använd hela COCO-datasetet istället för COCO128 för att förbättra modellens generaliseringsförmåga.
3. **Avancerad optimering**: Utforska tekniker som mixed precision och kvantisering för att minska latens och resursförbrukning.

---

## **Resultat och reflektion**
### **Resultat**
- Modellen uppnådde en genomsnittlig precision (mAP50) på 78% på COCO128, vilket är konkurrenskraftigt för datasetets storlek.
- Annoterade resultat sparades framgångsrikt och analyserades för att säkerställa modellens funktionalitet.
- Modellen exporterades till ONNX-format utan problem.

### **Reflektion**
Det här projektet har gett mig värdefulla insikter i att implementera och optimera avancerade objektigenkänningsmodeller. Jag ser stor potential för att vidareutveckla projektet genom att integrera mer avancerade tekniker och större dataset.

---

## **Framtida arbete**
1. **Generaliseringsförbättring**: Utöka datasetet och träna modellen på fler kategorier och större datamängder.
2. **Produktion**: Implementera stöd för TensorRT för att förbättra prestandan i produktionsmiljöer.
3. **Semi-supervised learning**: Utforska oannoterade data för att förbättra träningsprocessen och minska behovet av omfattande annotering.
