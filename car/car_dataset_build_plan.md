# Car Image Dataset Build Plan
**Make → Model → Generation classifier, designed for on-device iPhone deployment**
Version 1.0 · September 2026

---

## 0. Summary

This plan builds a merged, deduplicated, license-tagged image dataset for fine-grained car recognition. Every image is labeled hierarchically as **make → model → generation**, and the exact model year is kept as an extra label wherever a source provides it. The dataset combines about ten public sources with targeted collection to fill the biggest gaps: 2021–2026 model years and under-covered markets (Japan, India, Southeast Asia, Korea, the Middle East, Australia).

Two facts about the end goal shape almost every decision below. First, the final model will be a small network distilled from a large teacher, so the dataset needs to be broad and clean more than it needs to be enormous. Second, real-world inputs will be iPhone photos taken on streets and in parking lots, so the most important test set is one you shoot yourself.

### v1 targets

| Metric | Target |
|---|---|
| Canonical generation-level classes | 4,000–8,000 |
| Training images after per-class caps | ~0.8–1.5M |
| Per-class cap / floor (training) | 400 / 20 images |
| Held-out test sets | 4 (in-distribution, cross-source, surveillance, iPhone field set) |
| Images with recorded license + provenance | 100% |
| Label accuracy on a random 2,000-image human audit | ≥ 97% |

### Guiding principles

**Taxonomy before images.** Decide what the classes are before ingesting anything, or every source will pull the label space in a different direction. **Every image carries its provenance and license**, so you can export a commercially clean subset later and honor takedown requests. **Deduplicate before splitting**, because the web-scraped sources overlap and duplicates across train and test inflate every metric. **Split by vehicle, never by image**, since many sources have several near-identical photos of one car. **Balance sources within each class**, so the model learns cars rather than learning which dataset a photo came from. **The iPhone field test set is the scoreboard**: benchmark numbers on web photos are secondary.

---

## 1. Decisions to make up front

### 1.1 Research vs. commercial

Most of the public sources are non-commercial (CompCars, DVM-CAR, BoxCars116k) or have unclear licensing (Car-1000, the scraped sets). Rather than choosing now, build **one manifest** where every image has a `license_tier` column, and export two builds from it.

| Build | Contents | Use |
|---|---|---|
| `research` | Everything | Experiments, teacher training, benchmarking |
| `clean` | Wikimedia Commons (CC0 / public domain / CC BY / CC BY-SA), your own photos, anything with explicit written commercial permission | Anything you might ship |

The clean build is a whitelist, not a blacklist: an image is only included if its license is positively known to allow your use. A model trained on the research build should be treated as research-only, and that includes any student distilled from it.

### 1.2 Label granularity

The primary training label is the **generation** (e.g., Toyota Corolla E210, 2019–present). The exact model year is kept as an auxiliary label because some sources have it and it's free supervision, but the app should present generation plus year range. Facelifts (e.g., BMW "LCI" updates) can be an optional sub-level later; for v1, record them in the taxonomy but don't split classes on them.

**Rebadged twins** (Toyota GT86 / Subaru BRZ / Scion FR-S; Opel vs. Vauxhall; many GM/Holden/Chevrolet pairs) need a policy. Default: if the cars differ only by badge, merge them into one class that carries a list of badges; if they have visibly different grilles or lights, keep them separate.

### 1.3 Vehicle scope

For v1, include passenger cars, SUVs/crossovers, minivans, and light pickups. Exclude motorcycles, buses, heavy trucks, and commercial vans unless they come free with a source. In NHTSA's vPIC database these map to the vehicle types "Passenger Car", "Multipurpose Passenger Vehicle (MPV)" (which is where SUVs live), and "Truck" (filter to light-duty).

### 1.4 Compute and storage

Plan for roughly 150–250 GB of raw downloads and about 0.5 TB of working space once embeddings, crops, and exported builds are included. One modern GPU is enough for detection and CLIP embeddings over a few million images; in practice, disk I/O and JPEG decoding are the bottleneck, not the model.

---

## 2. Infrastructure

### 2.1 Directory layout

```
cars-dataset/
  raw/<source>/                 # untouched downloads, treated as read-only
    LICENSE.txt                 # copy of the source's license/terms, saved at download time
    RETRIEVAL.md                # URL, date, checksum, access notes
  manifests/
    images.parquet              # one row per image (schema below)
    taxonomy/
      makes.csv
      models.csv
      generations.csv
      aliases.csv
      rebadge_groups.csv
    raw_labels/<source>.parquet # each source's original label strings + counts
  derived/
    embeddings/clip_<model>.npy # row-aligned with images.parquet
    detections.parquet          # boxes, scores, primary-vehicle choice
  review/                       # queues exported for human review
  reports/                      # coverage heatmaps, per-source stats
  builds/<version>/{research,clean}/  # exported, sharded training sets
```

### 2.2 Manifest schema (`images.parquet`)

| Column | Type | Notes |
|---|---|---|
| `image_id` | str | SHA-256 of the original file bytes; stable primary key |
| `source` | str | e.g. `vmmrdb`, `dvm_car`, `commons`, `field` |
| `source_item_id` | str | The source's own filename or ID |
| `group_id` | str | Listing / vehicle / track ID; all photos of one physical car share it |
| `orig_path` | str | Path under `raw/` |
| `source_url` | str | Where the image came from, if known |
| `license` | str | e.g. `CC-BY-NC-4.0`, `CC-BY-SA-4.0`, `research-agreement`, `unknown` |
| `license_tier` | enum | `clean` / `research` / `excluded` |
| `attribution` | str | Required credit line for CC BY / BY-SA images |
| `raw_label` | str | Source's original label string, verbatim |
| `make_id`, `model_id`, `gen_id` | str | Canonical IDs; null where unresolved |
| `year` | int | Exact model year if the source gives it |
| `label_level` | enum | Deepest trustworthy level: `make` / `model` / `gen` / `year` |
| `label_origin` | enum | `source` / `auto` / `human` |
| `market` | str | Market the source represents: `US`, `UK`, `CN`, `BR`, ... |
| `viewpoint` | enum | `front` / `front_side` / `side` / `rear_side` / `rear` / `unknown` |
| `is_exterior` | bool | False for interiors, engine bays, detail shots |
| `bg_removed` | bool | True for DVM-CAR |
| `bbox` | list[int] | Primary vehicle box (x1, y1, x2, y2) in original pixels |
| `width`, `height` | int | Original dimensions |
| `phash` | str | 64-bit perceptual hash |
| `dup_cluster_id` | str | Shared by near-duplicates |
| `flags` | list[str] | e.g. `multi_vehicle`, `watermark`, `toy`, `render`, `low_res`, `review_pending` |
| `split` | enum | `train` / `val` / `test_id` / `test_xsrc` / `test_surv` / `test_field` / `unused` |

### 2.3 Tooling

| Need | Suggested tool |
|---|---|
| Manifest storage and queries | Parquet files queried with DuckDB |
| Bulk URL downloading | `img2dataset` |
| Perceptual hashing | `imagehash` |
| Embeddings and zero-shot filters | `open_clip` (a ViT-B/16 or ViT-L/14 checkpoint) |
| Vehicle detection for cropping | Any COCO-trained detector with car/truck classes (e.g., torchvision Faster R-CNN or RT-DETR) |
| Nearest-neighbor search | `faiss` |
| Visual browsing and QA | FiftyOne |
| Human review and relabeling | Label Studio |
| Versioning | DVC, or immutable build folders plus manifest hashes |

The detector is only used to produce crops for the dataset, so it never ships in the app and its licensing matters less here than for the on-device pipeline.

---

## 3. Phase 1: Build the taxonomy

**Goal:** a frozen, versioned list of makes, models, and generations that every source label can be mapped onto.

### Step 1.1: Pull the US make/model/year skeleton from vPIC

NHTSA's vPIC API is free, public-domain US government data covering model years 1981 onward. It gives you which make/model/year combinations actually existed in the US, which is the backbone for validating labels from VMMRdb, Stanford Cars, and The Car Connection.

For a full pull, prefer vPIC's downloadable standalone database over hammering the API; a naive loop over every make × year × vehicle type is tens of thousands of calls. For targeted lookups, the API is fine:

```python
import requests
from urllib.parse import quote

BASE = 'https://vpic.nhtsa.dot.gov/api/vehicles'
VTYPES = ['Passenger Car', 'Multipurpose Passenger Vehicle (MPV)', 'Truck']


def makes_for_type(vtype):
    r = requests.get(
        f'{BASE}/GetMakesForVehicleType/{quote(vtype)}',
        params={'format': 'json'},
        timeout=30,
    )
    r.raise_for_status()
    return [m['MakeName'] for m in r.json()['Results']]


def models_for(make, year, vtype):
    url = (
        f'{BASE}/GetModelsForMakeYear/make/{quote(make)}'
        f'/modelyear/{year}/vehicletype/{quote(vtype)}'
    )
    r = requests.get(url, params={'format': 'json'}, timeout=30)
    r.raise_for_status()
    return [m['Model_Name'] for m in r.json()['Results']]
```

vPIC lists hundreds of makes that are irrelevant here (trailer builders, kit-car shops, custom coachworks). Filter it down to makes that appear in at least one image source or on a curated list of roughly 150–400 consumer brands.

### Step 1.2: Add generations

vPIC has no concept of generations, and non-US markets need their own lineage anyway. The practical sources are Wikipedia model articles, whose infoboxes and "generations" sections list production years and chassis codes, and Wikidata, where individual generations are often separate items linked by manufacturer (P176) and follows / followed-by (P155 / P156). Coverage and consistency on Wikidata are uneven, so treat it as a candidate generator, not ground truth.

Use a semi-automatic process. A script proposes generation boundaries; a human verifies them for the top ~1,500 models ranked by image count, which covers the large majority of images. Long-tail models without a verified generation table stay at `label_level = model` rather than being bucketed by guesswork.

Generations sometimes differ by market (the US and European Corolla lines diverged for years; many Chinese-market cars are long-wheelbase variants). Where that happens, key the generation on (model, market).

`generations.csv`:

| Column | Example |
|---|---|
| `gen_id` | `toyota/corolla/e210` |
| `model_id` | `toyota/corolla` |
| `name` | `E210 (12th gen)` |
| `start_year`, `end_year` | `2018`, `null` (still in production) |
| `facelift_years` | `2022` |
| `markets` | `US;EU;CN;JP` |
| `body_styles` | `sedan;hatchback;wagon` |
| `source_ref` | Wikipedia/Wikidata URL |
| `verified` | `true` |

**Changeover years are ambiguous.** A new generation often launches mid-year, and "model year" rarely matches the calendar year, so a label like "2019 Corolla" can belong to either of two generations. When a source's year falls within one year of a generation boundary, set `gen_id` to null and keep the image at model level unless a human (or later, a trained teacher) resolves it.

### Step 1.3: Build the alias table

Before ingesting a single image, extract every distinct raw label string from every source's label list. These lists are small (tens of thousands of strings total) and mapping them first saves enormous rework.

`aliases.csv` maps a normalized raw string, optionally scoped to a source, onto canonical IDs. It absorbs abbreviations ("vw", "chevy", "merc"), market names (Vauxhall Astra → Opel Astra's canonical entry with a badge note), transliterated or Chinese-market names from CompCars and Car-1000, Portuguese naming conventions from BRCars, and each source's own formatting quirks (VMMRdb's `make_model_year` folder names, DVM-CAR's `brand-model-year-color` structure).

### Step 1.4: Define rebadge groups

`rebadge_groups.csv` lists groups of generations that are the same physical car under different badges, with a `visually_identical` flag (`yes` / `partial`). Apply the policy from Section 1.2: `yes` groups become one class carrying a badge list; `partial` groups stay separate.

### Step 1.5: Freeze taxonomy v1

After this point, taxonomy changes go through a changelog and a version bump, so manifests and builds always reference a known taxonomy version.

**Exit criteria for Phase 1:** every raw label string from every source either maps to a canonical ID or appears in an `unresolved.csv` with its image count, and at least 98% of images (by count) resolve.

---

## 4. Phase 2: Acquire the sources

**Send the access requests on day one.** CompCars requires a signed release agreement, BRCars requires an access request form, and Car-1000's license needs to be clarified with its authors. These can take days to weeks, and nothing else depends on them.

### 4.1 Source inventory

| # | Source | Access | Size | Labels | License tier | Role |
|---|---|---|---|---|---|---|
| 1 | **VMMRdb** | GitHub `faezetta/VMMRdb` (Kaggle mirror available) | 291,752 images, 9,170 classes | make / model / year, 1950–2016 | research | Core US and historical coverage |
| 2 | **DVM-CAR** | `deepvisualmarketing.github.io`, Figshare DOI 10.6084/m9.figshare.19586296 | 1,451,784 images at 300×300, 13.6 GB zip, 899 models | brand / model / year / color, plus ad, trim, and price tables | research (CC BY-NC) | Core UK/European 2000–2020; heavy subsampling |
| 3 | **CompCars** | Signed release agreement, CUHK MMLab | 136,726 web images (163 makes, 1,716 models) + ~50k surveillance fronts | make / model / year, boxes, viewpoints | research (non-commercial agreement) | China and global brands, ~2005–2016 |
| 4 | **Car-1000** | GitHub `toggle1995/Car-1000` → Google Drive / Baidu | 140,267 images, 1,000 models, 166 makes | model level only | research (license unstated; ask) | 2020s coverage, Chinese market |
| 5 | **Car Models 3778** | Hugging Face `Unit293/car_models_3887` | ~193k images at 512×512, 3,778 variants | make / model / year + 44 spec columns | research (license "other"; scraped from autoevolution.com) | Recent press photos with year labels, up to ~2023 |
| 6 | **BRCars** | Access request form via GitHub `danimtk/brcars-dataset` | ~300k images, 52k vehicles, 427 models | make / model / year | research | Latin American market |
| 7 | **BoxCars116k** | `medusa.fit.vutbr.cz` (Brno University of Technology) | 116,286 images, 27,496 vehicles, 693 classes | make / model / submodel / year, 3D boxes | research (CC BY-NC-ND; do not redistribute derivatives) | Surveillance viewpoints |
| 8 | **Stanford Cars** | Kaggle / Hugging Face mirrors | 16,185 images, 196 classes | make / model / year | research | **Test only**, never trained on |
| 9 | **The Car Connection** | Kaggle / GitHub | ~60k images | make / model / year + specs | research (scraped) | US 2010s press photos; top-up source |
| 10 | **Wikimedia Commons** | MediaWiki API | Varies by model; thousands of generation categories | Derived from categories | **clean** (CC0 / PD / CC BY / CC BY-SA only) | Commercially usable backbone; gap filling |
| 11 | **Your own iPhone photos** | You | 3,000–10,000 target | Labeled by you | **clean** | Field test set; clean training data |
| 12 | Regional listing sites | Only after terms-of-service and legal review | Varies | Seller-entered fields (noisy) | research unless licensed | Recent models in under-covered markets |

Deferred for v1: VeRi-776 and VehicleID (re-identification datasets with sparse model labels) and the kingjosephm Google-scraped set (its GitHub repo centers on scraping and training code; check whether images are distributed or whether you'd need to re-scrape).

### 4.2 Download hygiene

For every source, save the raw download untouched under `raw/<source>/`, record the retrieval URL, date, and SHA-256 checksum in `RETRIEVAL.md`, and save a copy of the license or agreement text at that moment. Licenses and hosting change, and you want a record of the terms you downloaded under.

### 4.3 Wikimedia Commons sub-plan

Commons is the only large source that's usable in a commercial product, so it deserves real effort. Many car models have per-generation categories (for example, a category for each Corolla generation), and those category names map directly onto your generation table.

The process: for each verified generation, find its Commons category; list files recursively to a shallow depth; skip subcategories whose names indicate non-exterior content (interiors, engines, dashboards, badges, wheels, details, advertisements, models/toys); fetch each file's license and author metadata; and keep only CC0, public domain, CC BY, and CC BY-SA files, storing the attribution string.

```python
import requests

API = 'https://commons.wikimedia.org/w/api.php'
HEADERS = {'User-Agent': 'CarDatasetBuilder/0.1 (contact: you@example.com)'}


def category_files(category):
    params = {
        'action': 'query',
        'list': 'categorymembers',
        'cmtitle': f'Category:{category}',
        'cmtype': 'file',
        'cmlimit': '500',
        'format': 'json',
    }
    while True:
        data = requests.get(API, params=params, headers=HEADERS, timeout=30).json()
        for m in data['query']['categorymembers']:
            yield m['title']
        if 'continue' not in data:
            break
        params.update(data['continue'])


def file_info(titles):  # up to 50 titles per call
    params = {
        'action': 'query',
        'titles': '|'.join(titles),
        'prop': 'imageinfo',
        'iiprop': 'url|size|mime|extmetadata',
        'format': 'json',
    }
    pages = requests.get(API, params=params, headers=HEADERS, timeout=30).json()[
        'query'
    ]['pages']
    for p in pages.values():
        ii = p['imageinfo'][0]
        meta = ii.get('extmetadata', {})
        yield {
            'title': p['title'],
            'url': ii['url'],
            'width': ii['width'],
            'height': ii['height'],
            'license': meta.get('LicenseShortName', {}).get('value'),
            'artist': meta.get('Artist', {}).get('value'),
        }
```

Follow Wikimedia's API etiquette: a descriptive User-Agent with contact info, serial rather than parallel requests, and backing off when asked. Note that CC BY-SA carries share-alike obligations; get a view on how those apply to trained model weights before shipping a commercial product that relies on BY-SA images.

### 4.4 Listing-site collection (optional, gated)

Only proceed after reviewing each site's terms of service, ideally with legal advice. If you go ahead: honor robots.txt, rate-limit aggressively, collect only the car photos and the structured vehicle fields (make, model, year, trim, listing ID), never store seller names, phone numbers, or addresses, and blur license plates at ingest. Prefer sites that offer data licensing or partner APIs. Treat seller-entered labels as noisy; expect a few percent to be wrong.

### 4.5 Field collection protocol (your iPhone photos)

This set does double duty: a fixed portion is your most important test set, and the rest is clean training data. Shoot in public places (streets, public parking lots) and ask permission before shooting on private property such as dealerships. For each car, take 3–6 photos from different angles (front, front three-quarter, side, rear three-quarter, rear) at normal phone distances, including some imperfect shots: glare, partial occlusion, dusk, rain. Give every photo of the same car the same `group_id` so they never straddle splits.

Label from visible badges and body cues at generation level, and have a second person check anything uncertain. At ingest, blur license plates and faces and strip GPS metadata from EXIF.

Aim for breadth over volume: a few photos each of many different generations beats hundreds of photos of common cars. Deliberately seek out cars from 2021–2026, EVs, and anything from under-covered markets you can find locally.

---

## 5. Phase 3: Ingest and normalize labels

**Goal:** every image from every source becomes one row in `images.parquet` with a canonical label, a group ID, and a license.

### Step 3.1: Write one adapter per source

Each adapter reads a source's raw files and emits manifest rows. Its jobs are to compute `image_id`, record `raw_label`, set `license` and `license_tier`, set `market`, and, most importantly, recover a **group ID** so that multiple photos of the same physical car stay together.

| Source | Where the group ID comes from |
|---|---|
| DVM-CAR | Advert ID from the dataset's image and ad tables |
| BoxCars116k | Vehicle/track ID |
| BRCars | Car instance ID |
| CompCars | None published; approximate by clustering (see below) |
| VMMRdb, Car-1000, Car Models 3778, The Car Connection | None; approximate by clustering |
| Commons | Uploader + date + category (one person's photo session of one car) |
| Field photos | Your per-car ID |

Where no group ID exists, approximate one: within each class, link images whose CLIP embeddings are very similar (a looser threshold than the deduplication step) and treat each connected cluster as a group. It won't be perfect, but it catches the most common case of several photos from one listing.

### Step 3.2: Resolve labels

Run each `raw_label` through three stages. Exact match against `aliases.csv` comes first. Then a normalized fuzzy match (lowercased, punctuation stripped, token-sorted), with anything above a strict similarity threshold auto-accepted and a middle band sent to review. Everything else lands in a review queue **sorted by image count**, so fixing one label string fixes thousands of images at a time.

Then map years to generations using `generations.csv`, applying the changeover-year rule from Step 1.2. Set `label_level` to the deepest level you trust for each image.

**Exit criteria for Phase 3:** fewer than 2% of images unresolved at model level, and every unresolved string is listed with its count.

---

## 6. Phase 4: Clean

### Step 4.1: Integrity checks

Decode every image and drop corrupt or truncated files. Drop images whose short side is below about 128 pixels, except for surveillance sources where small images are the point. Normalize color modes (CMYK, alpha channels) and apply EXIF orientation before anything else, then strip EXIF for privacy.

### Step 4.2: Detect and choose the primary vehicle

Run the detector on every image and keep car, truck, and bus boxes. Choose the primary vehicle as the largest, most central box. Drop images with no vehicle, or where the primary vehicle fills less than about 10% of the frame. Flag `multi_vehicle` when a second vehicle is at least half the primary's area, because then the label is ambiguous; send those to review or drop them. Store boxes in the manifest and generate crops only at export time, so you can change the crop margin later without reprocessing.

### Step 4.3: Content filters with CLIP zero-shot

Embed every image once with CLIP and reuse the embeddings for filtering, deduplication, grouping, and noise detection. Score each image against text prompts for content you don't want as exterior training data: car interiors, dashboards, engine bays, close-ups of wheels or badges, documents and keys (BRCars has these), toy or scale-model cars, 3D renderings and illustrations, and heavily modified or wrecked cars. Set `is_exterior = false` or add flags rather than deleting; interiors may be useful someday. Calibrate each prompt's threshold by eyeballing a few hundred images near the cutoff.

### Step 4.4: Tag viewpoints

CompCars ships viewpoint labels for its full-car images. Train a tiny viewpoint classifier on top of the CLIP embeddings using those labels, then apply it to every other source. Viewpoint tags drive balancing later and let you report accuracy by angle.

### Step 4.5: Deduplicate

Web-scraped sources overlap heavily, so run three passes. **Exact duplicates:** identical `image_id`. **Perceptual duplicates:** pHash Hamming distance of about 6 or less, which catches resizes and recompressions. **Semantic near-duplicates:** CLIP cosine similarity of about 0.95 or more, which catches crops and edits; tune the threshold by reviewing pairs at 0.92, 0.95, and 0.97.

Global all-pairs search over a few million embeddings is expensive, so either use a `faiss` IVF index or bucket by make and search within buckets (near-duplicates with different makes are rare, and pHash catches the exact cases globally).

```python
import faiss, numpy as np


def near_dup_pairs(emb, threshold=0.95, k=10):
    """emb: float32 array, L2-normalized, shape (n, d)"""
    index = faiss.IndexFlatIP(emb.shape[1])  # swap for an IVF index at scale
    index.add(emb)
    sims, nbrs = index.search(emb, k)
    for i in range(len(emb)):
        for s, j in zip(sims[i, 1:], nbrs[i, 1:]):
            if s >= threshold and i < j:
                yield i, j


# Union-find the pairs into dup_cluster_id, then keep one image per cluster.
```

Within each duplicate cluster, keep one image by priority: clean license tier first, then deeper label level, then higher resolution. **Duplicates with conflicting labels across sources are a gift**: they reveal label errors. Send them to review rather than silently picking one.

Finally, run an explicit check of Stanford Cars against everything else and remove any training image that duplicates a Stanford image, so the cross-source test set stays honest.

### Step 4.6: Hunt label noise

Two passes. Before any training: for each image, look at its 10 nearest CLIP neighbors; if most carry a different label, flag it for review. After the first teacher model is trained: send images where the teacher confidently disagrees with the label to review (the "confident learning" approach, implemented in libraries such as cleanlab). Review in order of expected impact, and track which sources produce the most errors.

### Step 4.7: Remove source shortcuts

Anything that identifies a source without identifying a car can become a shortcut the model learns instead of the car. The main culprits are watermarks and site logos (common on press-photo and listing sites), DVM-CAR's removed backgrounds, and license plates, whose format reveals the country and therefore the market. Flag watermarks with a CLIP prompt or OCR, and crop them out or drop the image. Keep `bg_removed` so that training can composite DVM-CAR cars onto real backgrounds. Blur license plates at export for every source; that removes the regional shortcut and handles privacy at the same time.

---

## 7. Phase 5: Coverage analysis and gap filling

### Step 5.1: Generate coverage reports

After cleaning, produce these reports under `reports/`: a histogram of images per class; a list of classes below the 20-image floor; a heatmap of image counts by market × decade; the viewpoint distribution per class; and the number of distinct sources per class. Single-source classes are a shortcut risk and deserve priority even if their image counts look healthy.

### Step 5.2: Prioritize gaps

Expect the biggest holes to be 2021–2026 model years in every market except China, Japanese domestic models (kei cars in particular), India, Southeast Asia, Korea's home market, the Middle East, Australia, pre-1980 classics, and newer EV brands. Rank gap classes by how often you expect users to photograph them; a missing current-generation Toyota RAV4 matters far more than a missing 1960s coachbuilt coupe.

### Step 5.3: Fill in order of license cleanliness

Fill from Wikimedia Commons first, then your own photos, then any licensed or partner data, and only then listing sites that have passed review. The target is every class at or above the floor, drawn from at least two sources wherever possible.

This phase never really ends: new model years arrive every year. The on-device design (a cosine classifier with imprinted class prototypes) lets you add a new generation from 20–50 photos between full retrains.

---

## 8. Phase 6: Splits and sampling

**Split first, then sample.** Assign splits at the group level before any capping, so no car ever appears in two splits.

### 8.1 Evaluation sets

| Split | Contents | Purpose |
|---|---|---|
| `val` | ~5% of groups per class (at least 1 group, up to ~20 images per class) | Tuning and early stopping |
| `test_id` | Same construction, disjoint from `val` | In-distribution accuracy |
| `test_xsrc` | All of Stanford Cars, mapped to canonical classes | Cross-source generalization on clean web photos |
| `test_surv` | A held-out slice of BoxCars116k vehicles | Surveillance and odd-angle robustness |
| `test_field` | A fixed portion of your iPhone photos (at least 1,500–2,000 images, never trained on) | **The headline metric** |

Report every metric at make, model, and generation level, and slice results by market, decade, viewpoint, and head vs. tail classes.

### 8.2 Training-set sampling

Within the training split, cap each class at 400 images. Fill each class's cap round-robin across sources, and within a source take one image per group before taking a second from any group, spreading picks across viewpoints.

Source-specific rules:

| Source | Rule |
|---|---|
| DVM-CAR | At most ~60 images per model-year and at most 2 views per advert; composite onto real backgrounds during training |
| CompCars surveillance | At most ~15% of any class's cap |
| BoxCars116k | At most 2 images per vehicle |
| Car-1000 | Take all; supervises make and model levels only |
| VMMRdb | Take all images in rare classes; cap the popular ones |
| Scraped top-up sources | Only to reach the floor in classes that can't reach it otherwise |

Classes that still fall below the floor after gap filling roll up to model level and train only the make and model outputs.

Capping sets the dataset's composition; training-time balancing (sampling classes in proportion to the square root of their counts, or a class-balanced loss) is a separate knob handled in the training code.

### 8.3 Export format

Export each build (`research` and `clean`) as WebDataset tar shards for fast streaming during training. Each record holds the crop (primary vehicle box with a small margin, longest side capped at 512 pixels) and a JSON sidecar with `image_id`, all label levels, `viewpoint`, `bg_removed`, and `source`. Ship alongside it a `classes.json` containing the full hierarchy and the class-index mapping, plus the taxonomy version.

---

## 9. Phase 7: QA, documentation, versioning

**Audit.** Draw 2,000 random training images and have a person check each label at generation level. Compute accuracy overall and per source. If any source falls below about 90%, tighten its filters or drop it, and re-audit.

**Dataset card.** Write a short document covering sources and their licenses, counts per split and per market, the taxonomy version, known biases (market skew toward the US, UK, and China; press-photo vs. street-photo mix), and audit results.

**Versioning.** Builds are immutable once exported. Name them `v1.0`, `v1.1`, and so on, record the manifest hash and taxonomy version in each, and keep a changelog of what changed and why.

**Takedowns.** Because every row has `source_url` and `image_id`, you can remove any image on request and re-export.

---

## 10. Timeline

Rough estimates for one person working full-time; double them for part-time. Phases overlap, especially acquisition (which involves waiting on access approvals) and gap filling (which continues indefinitely).

| Phase | Estimate | Notes |
|---|---|---|
| 0. Decisions and infrastructure | 2–3 days | Send access requests on day one |
| 1. Taxonomy | 1–2 weeks | Generation verification is the long pole |
| 2. Acquisition | 1–2 weeks | Mostly waiting and downloading; overlaps Phase 1 |
| 3. Ingest and label normalization | 1–2 weeks | Alias review sorted by image count |
| 4. Cleaning | ~2 weeks | Embeddings, filters, dedupe, first noise pass |
| 5. Coverage and gap filling | 2–4 weeks for v1 | Commons pipeline plus field shooting |
| 6–7. Splits, export, QA | ~1 week | Includes the 2,000-image audit |
| **Total for v1** | **~8–12 weeks** | |

---

## 11. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| CompCars or BRCars access is slow or denied | Less China / Latin America coverage | Proceed without them; both are additive, and Car-1000 partly covers China |
| Taxonomy work balloons | Weeks lost verifying obscure generations | Verify only the top ~1,500 models by image count; the long tail stays at model level |
| Model learns sources instead of cars | Great benchmarks, poor field accuracy | Round-robin sampling, background compositing, watermark and plate removal, cross-source and field test sets |
| Test leakage through web duplicates | Inflated metrics | Dedupe before splitting; explicit Stanford Cars duplicate check |
| Label noise from scraped and seller-entered data | Lower ceiling, confused classes | kNN flagging, teacher-disagreement review, per-source audit thresholds |
| Non-commercial data leaks into a shipped product | Legal exposure | Per-image `license_tier`; clean build is a whitelist; research-trained models stay research-only |
| Recent models underrepresented | App fails on the cars people see most | Commons and field collection prioritized by expected user frequency; prototype imprinting between retrains |
| Changeover-year labels assigned to the wrong generation | Systematic generation errors | Boundary-year rule sets `gen_id` to null pending review |

---

## 12. Definition of done (v1)

| Item | Done when |
|---|---|
| Taxonomy v1 | Frozen, versioned, ≥ 98% of images resolve |
| Manifest | Every image has `image_id`, `group_id`, canonical label, `label_level`, `license_tier` |
| Cleaning | Integrity, detection, content filters, viewpoint tags, and dedupe complete; flagged items reviewed or excluded |
| Coverage | Coverage reports generated; every class at or above floor or rolled up to model level |
| Splits | Group-level splits assigned; all five evaluation sets built; Stanford duplicate check passed |
| Field test set | At least 1,500–2,000 labeled iPhone photos, plates and faces blurred, never used in training |
| Builds | `research` and `clean` exported as WebDataset shards with `classes.json` |
| QA | 2,000-image audit at ≥ 97% accuracy; dataset card written |
