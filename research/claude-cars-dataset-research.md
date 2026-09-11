# Car Image Datasets for Make/Model/Year Classification: A Deep-Dive Evaluation

## TL;DR
- **No single public dataset cleanly solves "recent make + model + exact year."** For research/hobby use, the best combination is **Stanford Cars (Cars196)** as a clean benchmark plus **VMMRdb** (US, 1950–2016) and **CompCars** (Chinese market) for scale and year coverage; for recent vehicles (2016→2026) you will almost certainly need to build a custom dataset by scraping car-listing sites and using **NHTSA vPIC** for the make/model/year taxonomy.
- **True exact-year labels are rare and visually hard.** Most datasets label a "year" that really means a generation/facelift window; adjacent model years of the same generation are frequently visually identical, so exact-year prediction is often ill-posed. Datasets with genuine year/generation labels: Stanford Cars, CompCars, VMMRdb, BoxCars116k, DVM-CAR, BRCars.
- **Licensing is the key commercial filter.** Stanford Cars, CompCars, DVM-CAR (CC BY-NC), and BoxCars116k (CC BY-NC-ND) are research/non-commercial only. For commercial use, build your own dataset (respecting site ToS and image copyright) or use permissively-licensed checkpoints, and lean on NHTSA vPIC (public-domain US government data) for the label taxonomy.

## Key Findings

### The core tension: "year" is the hard part
Make and model recognition is a mature, largely solved fine-grained classification problem (top models reach ~97% on Stanford Cars). But the user's requirement for **year** is where both the datasets and the underlying visual signal break down:
- Automakers keep a body/design essentially fixed across a generation (typically 6–8 years) with a mid-cycle "facelift" (usually around year 3–4, e.g., BMW's "LCI"). Within a generation, consecutive model years are often visually indistinguishable without a VIN or registration record.
- Consequently, most "year" labels in car datasets are best interpreted as **generation/facelift** labels, not exact calendar-year labels. Stanford Cars, for example, pins many classes to a single year (e.g., "2012 BMW M3 coupe") even though that generation spans multiple years.
- Practical implication: treat "year" as **generation** (or a year-range) rather than exact year, and expect exact-year accuracy to have an intrinsic ceiling below make/model accuracy. Commercial photo-ID services (e.g., Car Reveal) explicitly caveat that "an exact model year usually requires a VIN or registration record."

### Datasets that actually carry year/generation labels
- **Stanford Cars (Cars196):** make+model+year in the class string, but only ~2012-era and older.
- **CompCars:** hierarchical make → model → year of manufacture (the classification subset spans roughly 2005–2016).
- **VMMRdb:** make+model+production year, models 1950–2016.
- **BoxCars116k:** make+model+submodel+model year (surveillance viewpoints).
- **DVM-CAR:** brand-model-year-color folder structure, UK market ~2000–2020.
- **BRCars:** make → model → year → instance hierarchy, Brazilian market.

### Availability alert: Stanford Cars original link is dead
The original Stanford host (`ai.stanford.edu/~jkrause/cars/`) is **down**, and `torchvision.datasets.StanfordCars` no longer downloads it — the class remains but raises `"The original URL is broken so the StanfordCars dataset cannot be downloaded anymore,"` and the `download=True` flag is a no-op kept only for backward compatibility. The data survives via mirrors: **Kaggle** (jutrera, "Stanford Car Dataset by classes folder"), **Hugging Face** (`tanganke/stanford_cars`, `HuggingFaceM4/Stanford-Cars`, `Multimodal-Fatima/StanfordCars_train`), and **Roboflow Universe**.

## Details

### Comparison table

| Dataset | # Images | # Classes | Granularity | Vehicle years | Image source / viewpoints | Extra annotations | License | Availability | Known issues |
|---|---|---|---|---|---|---|---|---|---|
| **Stanford Cars (Cars196)** | 16,185 (8,144 train / 8,041 test) | 196 | Make+Model+Year | ~1991–2012 (mostly 2012) | Web photos (auctions, dealers, classifieds); multi-view | Bounding boxes | Non-commercial research (described variously as "non-commercial research" or "unknown") | Original site down; mirrors on Kaggle/HF/Roboflow | Old vehicles; small; a known test-label duplication error; near-saturated |
| **CompCars** | 208,826 total (136,726 web + 44,481 surveillance) | 1,716 models / 163 makes (431-model classification subset) | Make+Model+Year | ~2005–2016 | Web + surveillance; front/rear/side/interior | Bounding boxes, viewpoint, car parts (~27k), attributes | Research/academic only | Available on request from CUHK | Chinese-market skew; biased default train/test split |
| **VMMRdb** | 291,752 | 9,170 | Make+Model+Year | 1950–2016 | Craigslist user photos (US); varied devices/angles | User-added boxes on subsets | Research (varies) | GitHub (faezetta/VMMRdb), Kaggle mirror | Severe class imbalance, label noise, long tail |
| **BoxCars116k** | 116,286 (27,496 vehicles) | 693 fine-grained | Make+Model+Submodel+Year | modern (surveillance era) | Traffic surveillance cameras; high-elevation angles | 3D bounding boxes, foreground masks, viewpoint | CC BY-NC-ND 4.0 | medusa.fit.vutbr.cz download; GitHub code | Surveillance-only viewpoints; small images; no interior |
| **DVM-CAR (2.0)** | 1,451,784 | 899 models | Brand-Model-Year-Color | ~2000–2020 (UK) | UK classified ads; 8 viewpoints; backgrounds removed | Sales/price/trim tables, body type | CC BY-NC | deepvisualmarketing.github.io; Figshare DOI 10.6084/m9.figshare.19586296 | Non-commercial; UK-market; backgrounds stripped |
| **BRCars** | ~300k (427-model); 212,609 (196-model set) | 427 / 196 | Make→Model→Year→Instance | Brazilian market (recent) | Brazilian ad site; exterior + interior; unstandardized perspective | Perspective labels (CLIP-assisted) | Research | Via authors / SIBGRAPI 2021 paper | Imbalanced; interior/exterior mixed; noise (keys, docs) |
| **VeRi-776** | ~50k (49,357) | 776 identities | Vehicle ID (+ type, color, brand attrs) | modern | 20 surveillance cameras | BBoxes, color, type, plate, spatio-temporal | Research | GitHub (VehicleReId/VeRi) | Re-ID, not MMY; limited model labels |
| **VehicleID (PKU)** | 221,763 | 26,267 identities (~10k model-labeled) | Vehicle ID (+ partial model) | modern | Surveillance; front/rear only | Model labels on subset | Research | Peking University | Re-ID focus; only front/rear views |
| **The Car Connection (Gervais)** | ~60,000 | make/model/year (US) | Make+Model+Year (+specs) | broad US market | thecarconnection.com press photos | price, HP, body style, specs | Scraped (ToS/copyright caution) | GitHub + Kaggle mirror | Single-site scrape; duplicates; studio-style images |
| **kingjosephm vehicle_make_model** | ~700k | 574 make-models (8,274 make-model-category-year combos) | Make+Model+Category+Year | US market | Google Images scrape | none | Scraped (ToS/copyright caution) | GitHub | Google-scraped label noise; drops exotics/EV startups |
| **Indian Vehicle datasets (DataCluster etc.)** | 35k–50k+ | vehicle type mostly | Type (not MMY) | recent | Street/mobile India | boxes | Mixed | Kaggle/GitHub | Not make/model/year |

### Regional and newer datasets
- **CompCars** (China) remains the largest well-annotated make/model/year web dataset outside the US.
- **DVM-CAR** (UK) has the largest image count of any car MMY dataset and is best for recent (up to ~2020) European-market vehicles, but is CC BY-NC.
- **BRCars** (Brazil) is notable for including interior images and being drawn from ad listings, mirroring real-world "messy" data.
- **MPF-Cars** (335,011 images, 2,019 models, 180 manufacturers) and **DeepCar 5.0** (headlight/grille/bumper analysis for recognition) are recent additions cited in the 2023–2026 literature.
- **Car-333** (157,023 training images, 333 categories) is an older web-scraped alternative.
- **Indian** datasets (DataCluster Labs Indian Vehicle Dataset ~35–50k; Roboflow "Indian Vehicle Classification"; Bharatiya Vehicle Dataset) are mostly vehicle *type* (car/bike/truck/rickshaw), not make/model/year — of limited use for fine-grained MMY.
- **Thai** datasets (VTID2 ~4,356 images of 5 vehicle types; VMID ~2,072 logo images of 11 brands) cover type and make logos, not model/year.
- **AIDOVECL** is an AI-generated (outpainted) vehicle dataset on Hugging Face (CC BY-4.0, DOI 10.57967/hf/8444) for eye-level classification/localization.
- **Roboflow Universe** hosts several community "Car Make Model Year" datasets (e.g., Senior Design's 196-class ~10k-image set, CC BY 4.0) that mirror or extend Stanford Cars.

### Vehicle taxonomy / API sources for labels
- **NHTSA vPIC** (`vpic.nhtsa.dot.gov/api`): free, public US government API providing all makes, models by year, vehicle type, and VIN decoding for **model years 1981 and forward**. It is available 24/7, free, and requires no registration (only an automated rate limit). Also downloadable as standalone databases. This is the canonical source for a US make/model/year label taxonomy and for normalizing scraped labels. It is public-domain US government data — usable commercially.
- Commercial VIN/taxonomy APIs and marketplace scrapers (Apify actors for Autotrader/cars.com/CarGurus, etc.) exist but carry ToS and cost considerations.

### Scraping listing sites: legal/ToS considerations
- Used-car marketplaces (Autotrader US/UK, cars.com, CarGurus, Craigslist) provide make/model/year-labeled images from dealer/private listings and are the practical route to recent vehicles.
- Legality is nuanced: scraping publicly available, non-personal data is generally treated as permissible in the US (following the *hiQ v. LinkedIn* line of cases), but **each site's Terms of Service typically prohibit automated scraping**, the images are copyrighted by the lister/photographer, and personal data (seller contact info) must be avoided.
- Existing scraped datasets (The Car Connection; kingjosephm's ~700k Google-scraped set) exist as precedent but redistribute copyrighted images and are best treated as research-only.
- Safer options: (a) scrape only labels + your own captured images; (b) use datasets whose licenses permit your use; (c) obtain data via official APIs/partnerships; (d) get legal review before any commercial deployment.

### State-of-the-art baselines and pretrained checkpoints
- **Stanford Cars top-1 accuracy** now sits around **96–97%** and is effectively near-saturated. The best well-documented single-model result is **CMAL-Net with a TResNet-L backbone at 97.1%** (Liu et al., *Pattern Recognition* vol. 140, art. 109550, 2023; the same method scores 94.9% with a ResNet-50 backbone). Strong "plain" fine-tunes land ~94–96%: DenseNet-161 94.6%, EfficientNet-B7 94.7%, TResNet-L 96.0%, TransFG/ViT ~94.8%, API-Net and DCAL ~95.3%. Claims of 99%+ are outliers and should be treated with caution.
- **CompCars** fine-grained accuracy exceeds 90% on the biased default split but drops to a realistic **~61%** (ResNet-50) on the corrected re-split of Buzzelli & Segantin (*Sensors* 2021) — a caution about over-optimistic benchmark numbers.
- **VMMRdb-3036** (make+model+year, 3,036 classes): ResNet-50 achieves **51.76% top-1 / 92.90% top-5** — a realistic picture of how hard fine-grained year-level classification is at scale.
- **Pretrained checkpoints on Hugging Face:** `therealcyberlord/stanford-car-vit-patch16` (ViT, ~86% on the Stanford Cars test split, Apache-2.0) and `Jordo23/vehicle-classifier` (EfficientNet-B4, 8,949 make/model/year classes trained on VMMRdb, MIT license — explicitly free for commercial use). Both are good fine-tuning starting points; the MIT-licensed one is the more commercially safe base.

## Recommendations

### Stage 1 — Establish a baseline (research/hobby)
1. Start with **Stanford Cars** (Kaggle or Hugging Face mirror) to validate your pipeline and get a make+model+year model on a clean 196-class benchmark. Fine-tune a strong ImageNet/CLIP backbone (ViT-B/16, ConvNeXt, or EfficientNet); expect ~90–96% top-1. Warm-start from `therealcyberlord/stanford-car-vit-patch16`.
2. Add **VMMRdb** for US coverage and far more classes/years (1950–2016). Prune long-tail classes (<50–100 images) and use stratified sampling to combat the severe imbalance. `Jordo23/vehicle-classifier` (VMMRdb-trained, MIT) is a strong checkpoint to fine-tune from.

### Stage 2 — Scale and add recency
3. For **recent (2016→2026) vehicles**, no benchmark suffices — build a custom set. Enumerate the label space with **NHTSA vPIC** (add European/Asian taxonomies as needed), then collect images per make/model/year. For non-US recency, **DVM-CAR** (up to ~2020, UK) and **CompCars/BRCars** extend coverage — but respect their non-commercial licenses.
4. Treat "year" as **generation/facelift** buckets, not exact years. Group model years within a generation unless you have evidence they're visually separable; expose a year-range in the output. Add a VIN/registration fallback where exact-year output is required.

### Stage 3 — Commercial deployment
5. **Do not ship a model trained on non-commercial data** (Stanford Cars, CompCars, DVM-CAR, BoxCars) in a commercial product without clearing rights. Instead: (a) build a proprietary dataset from licensed/owned imagery or ToS-cleared sources; (b) use **vPIC** (public domain) for labels; (c) base models on permissive checkpoints (e.g., the MIT-licensed `Jordo23` model); (d) get legal review of any scraping.
6. Budget for continuous data refresh: new model years appear annually and facelifts shift visual features, so plan a recurring scrape/label/retrain loop.

### Benchmarks/thresholds that change the plan
- If **make+model** (not year) is acceptable → Stanford Cars + VMMRdb alone likely suffice; skip custom scraping.
- If you need **exact year** and accuracy stalls well below make/model accuracy → that is expected (cf. VMMRdb-3036's 51.8% top-1); fall back to generation labels or add VIN/metadata.
- If **commercial** → licensing, not accuracy, is the binding constraint; prioritize vPIC + owned/licensed imagery from day one.

## Caveats
- **"Year" ≠ exact year in most datasets.** Labels usually encode a representative year or a generation; adjacent same-generation years are often visually identical. Manage expectations accordingly.
- **Benchmark accuracy is optimistic.** CompCars' 90%+ collapses to ~61% on a realistic re-split; Stanford Cars is near-saturated and small, so its high numbers won't transfer to messy real-world (surveillance, phone) images or to classes/years absent from training.
- **Licensing is often ambiguous or restrictive.** Stanford Cars' license is variously described as "non-commercial research" or "unknown"; CompCars, DVM-CAR (CC BY-NC), and BoxCars (CC BY-NC-ND) are clearly non-commercial. Roboflow/Kaggle re-hosts sometimes assign CC BY 4.0 to derivatives, but the underlying images may carry different rights.
- **Scraping is legally gray.** Public-data scraping is often defensible, but site ToS commonly forbid it and images are copyrighted; avoid personal data and obtain legal advice for commercial use.
- **Regional coverage is uneven.** US (VMMRdb), China (CompCars), UK (DVM-CAR), Brazil (BRCars) are covered; India/SE-Asia datasets are mostly vehicle-type, not MMY. A globally robust classifier requires combining sources and normalizing overlapping make/model taxonomies.
- **Figures vary across sources.** CompCars image/model counts, VeRi image totals, and VMMRdb subset sizes differ slightly between papers due to different subsets/versions; where sources conflicted, the most commonly cited values are reported here.