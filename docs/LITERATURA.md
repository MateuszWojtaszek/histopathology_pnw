# Literatura — co i w jakiej kolejności czytać

Lista ułożona pod zadania Etapu 1 od promotora (B1–B3 badawcze, I1–I3 implementacyjne).
Bloki czytaj w podanej kolejności: najpierw to, co jest potrzebne do kodu (I1, I3), potem
przegląd literatury (B1, B2), na końcu tło na później.

★ = obowiązkowe, ☆ = uzupełniające (przejrzeć abstrakt, wstęp i wyniki).
Wszystkie DOI i identyfikatory arXiv sprawdzone w Crossref / arXiv (2026-10-15).

---

## 1. Model i metryki (najpierw — potrzebne do I1 i I3)

| # | | Pozycja | Po co |
|---|---|---|---|
| 1.1 | ★ | Stringer C., Wang T., Michaelos M., Pachitariu M., *Cellpose: a generalist algorithm for cellular segmentation*, Nature Methods, 2021. [doi:10.1038/s41592-020-01018-x](https://doi.org/10.1038/s41592-020-01018-x) | Flow fields i to, jak z nich powstają maski instancji. Bez tego nie zrozumiesz wyjścia modelu. |
| 1.2 | ★ | Kirillov A. i in., *Segment Anything*, ICCV, 2023. [doi:10.1109/ICCV51070.2023.00371](https://doi.org/10.1109/ICCV51070.2023.00371) · [arXiv:2304.02643](https://arxiv.org/abs/2304.02643) | Encoder ViT-L, z którego startuje student. Wystarczy sekcja o architekturze. |
| 1.3 | ★ | Pachitariu M., Rariden M., Stringer C., *Cellpose-SAM: superhuman generalization for cellular segmentation*, bioRxiv, 2025. [doi:10.1101/2025.04.28.651001](https://doi.org/10.1101/2025.04.28.651001) | Architektura oracle/studenta, augmentacje, **lista danych treningowych** (MoNuSeg, PanNuke… → zależność wyników oracle). |
| 1.4 | ★ | Pachitariu M., Stringer C., *Cellpose 2.0: how to train your own model*, Nature Methods, 2022. [doi:10.1038/s41592-022-01663-4](https://doi.org/10.1038/s41592-022-01663-4) | Trenowanie na małej liczbie adnotacji i human-in-the-loop, czyli to, co robi I3. |
| 1.5 | ★ | Kirillov A., He K., Girshick R., Rother C., Dollár P., *Panoptic Segmentation*, CVPR, 2019. [doi:10.1109/CVPR.2019.00963](https://doi.org/10.1109/CVPR.2019.00963) · [arXiv:1801.00868](https://arxiv.org/abs/1801.00868) | Definicja PQ = DQ × SQ i dowód, że dopasowanie przy IoU > 0.5 jest jednoznaczne. Podstawa metryki z I1. |
| 1.6 | ★ | Graham S. i in., *HoVer-Net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images*, Medical Image Analysis, 2019. [doi:10.1016/j.media.2019.101563](https://doi.org/10.1016/j.media.2019.101563) | PQ w segmentacji jąder, krytyka Dice/AJI, opis zbioru CoNSeP. |
| 1.7 | ★ | Foucart A., Debeir O., Decaestecker C., *Panoptic quality should be avoided as a metric for assessing cell nuclei segmentation and classification in digital pathology*, Scientific Reports, 2023. [doi:10.1038/s41598-023-35605-7](https://doi.org/10.1038/s41598-023-35605-7) | Ograniczenia PQ dla małych obiektów. Do dyskusji wyników w I1. |

Kod do przejrzenia razem z 1.1–1.4 (zainstalowany pakiet w `.venv`): `cellpose/vit.py`
(`CPSAM`, `load_pretrained`), `cellpose/models.py` (`CellposeModel.eval`), `cellpose/train.py`
(`train_seg`), `cellpose/metrics.py` (`average_precision`, `aggregated_jaccard_index`).

---

## 2. B3 — zbiory benchmarkowe

| # | | Pozycja | Po co |
|---|---|---|---|
| 2.1 | ★ | Kumar N. i in., *A Dataset and a Technique for Generalized Nuclear Segmentation for Computational Pathology*, IEEE TMI, 2017. [doi:10.1109/TMI.2017.2677499](https://doi.org/10.1109/TMI.2017.2677499) | Oryginalny MoNuSeg i definicja AJI. |
| 2.2 | ★ | Kumar N. i in., *A Multi-Organ Nucleus Segmentation Challenge*, IEEE TMI, 2020. [doi:10.1109/TMI.2019.2947628](https://doi.org/10.1109/TMI.2019.2947628) | Challenge MoNuSeg 2018: oficjalny split i wyniki odniesienia. |
| 2.3 | ★ | Gamper J. i in., *PanNuke: An Open Pan-Cancer Histology Dataset for Nuclei Instance Segmentation and Classification*, ECDP 2019 (LNCS). [doi:10.1007/978-3-030-23937-4_2](https://doi.org/10.1007/978-3-030-23937-4_2) | Opis PanNuke. |
| 2.4 | ☆ | Gamper J. i in., *PanNuke Dataset Extension, Insights and Baselines*, arXiv, 2020. [arXiv:2003.10778](https://arxiv.org/abs/2003.10778) | Rozszerzona wersja PanNuke, foldy, baseline'y. |
| 2.5 | ★ | Graham i in., 2019 (HoVer-Net, poz. 1.6) | Źródło CoNSeP: nie ma osobnej publikacji. |
| 2.6 | ★ | Litjens G. i in., *1399 H&E-stained sentinel lymph node sections of breast cancer patients: the CAMELYON dataset*, GigaScience, 2018. [doi:10.1093/gigascience/giy065](https://doi.org/10.1093/gigascience/giy065) | Opis danych CAMELYON16/17. |
| 2.7 | ☆ | Bándi P. i in., *From Detection of Individual Metastases to Classification of Lymph Node Status at the Patient Level: The CAMELYON17 Challenge*, IEEE TMI, 2019. [doi:10.1109/TMI.2018.2867350](https://doi.org/10.1109/TMI.2018.2867350) | Zadanie i etykiety CAMELYON17. |
| 2.8 | ☆ | Graham S. i in., *Lizard: A Large-Scale Dataset for Colonic Nuclear Instance Segmentation and Classification*, ICCV Workshops, 2021. [arXiv:2108.11195](https://arxiv.org/abs/2108.11195) | Lizard/CoNIC zawiera m.in. CoNSeP i PanNuke, co ma znaczenie przy pytaniu o niezależność oracle. |

Porównanie do uzupełnienia w trakcie czytania (pola z `?` sprawdzić w źródłach):

| Zbiór | Obrazy | Adnotacje | Split | W danych treningowych Cellpose-SAM? |
|---|---|---|---|---|
| MoNuSeg | kafle 1000×1000, 40×, TCGA, wiele narządów; na dysku 37 train + 14 test | poligony instancji (XML) | oficjalny train/test | **tak** |
| PanNuke | łaty 256×256, 19 typów tkanek | maski instancji + 5 klas jąder | 3 foldy | **tak** |
| CoNSeP | kafle 1000×1000, 40×, jelito grube; liczba obrazów ? | maski instancji + klasy | train/test ? | ? (pośrednio przez CoNIC/Lizard — sprawdzić w 1.3) |
| CAMELYON17 | WSI węzłów chłonnych, 5 ośrodków | etykiety slajdów/pacjentów, część obrysów zmian; **brak masek jąder** | challenge train/test | ? |

Wniosek dla I1: w MoNuSeg i PanNuke metryki oracle są zawyżone, bo model widział te dane.
Należy to wprost napisać przy raportowaniu wyników.

---

## 3. B1 — AL ogólnie i w segmentacji obrazów medycznych

Kolejność: przeglądy → metody bazowe (te same, z którymi porównuje się IT2 FLS) → jak poprawnie ewaluować AL.

| # | | Pozycja | Po co |
|---|---|---|---|
| 3.1 | ★ | Settles B., *Active Learning Literature Survey*, Computer Sciences Technical Report 1648, University of Wisconsin–Madison, 2009. [PDF](https://research.cs.wisc.edu/techreports/2009/TR1648.pdf) | Terminologia: pool-based, query strategy, uncertainty/diversity. |
| 3.2 | ★ | Ren P. i in., *A Survey of Deep Active Learning*, ACM Computing Surveys, 2021. [doi:10.1145/3472291](https://doi.org/10.1145/3472291) | Deep AL: batch selection, cold start, problemy z sieciami. |
| 3.3 | ★ | Budd S., Robinson E.C., Kainz B., *A survey on active learning and human-in-the-loop deep learning for medical image analysis*, Medical Image Analysis, 2021. [doi:10.1016/j.media.2021.102062](https://doi.org/10.1016/j.media.2021.102062) | AL w obrazowaniu medycznym, koszt adnotacji, człowiek w pętli. |
| 3.4 | ★ | Wang H. i in., *A comprehensive survey on deep active learning in medical image analysis*, Medical Image Analysis, 2024. [doi:10.1016/j.media.2024.103201](https://doi.org/10.1016/j.media.2024.103201) · [lista prac + kod](https://github.com/LightersWang/Awesome-Active-Learning-for-Medical-Image-Analysis) | Najnowszy przegląd. Repo z listą prac jest punktem wyjścia do szukania dalej. |
| 3.5 | ★ | Yang L., Zhang Y., Chen J., Zhang S., Chen D.Z., *Suggestive Annotation: A Deep Active Learning Framework for Biomedical Image Segmentation*, MICCAI, 2017. [doi:10.1007/978-3-319-66179-7_46](https://doi.org/10.1007/978-3-319-66179-7_46) | Klasyka AL w segmentacji biomedycznej (niepewność + reprezentatywność). |
| 3.6 | ★ | Nath V., Yang D., Landman B.A., Xu D., Roth H.R., *Diminishing Uncertainty Within the Training Pool: Active Learning for Medical Image Segmentation*, IEEE TMI, 2021. [doi:10.1109/TMI.2020.3048055](https://doi.org/10.1109/TMI.2020.3048055) | AL w segmentacji medycznej z porównaniem strategii zapytań. |
| 3.7 | ☆ | Zheng H. i in., *Biomedical Image Segmentation via Representative Annotation*, AAAI, 2019. [doi:10.1609/aaai.v33i01.33015901](https://doi.org/10.1609/aaai.v33i01.33015901) | Wybór reprezentatywnych próbek bez niepewności (alternatywa dla 3.5). |
| 3.8 | ★ | Gal Y., Ghahramani Z., *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*, ICML, 2016. [arXiv:1506.02142](https://arxiv.org/abs/1506.02142) | Teoria MC-Dropout. |
| 3.9 | ★ | Gal Y., Islam R., Ghahramani Z., *Deep Bayesian Active Learning with Image Data*, ICML, 2017. [arXiv:1703.02910](https://arxiv.org/abs/1703.02910) | MC-Dropout jako selektor AL (BALD, entropia). Metoda porównawcza. |
| 3.10 | ★ | Sener O., Savarese S., *Active Learning for Convolutional Neural Networks: A Core-Set Approach*, ICLR, 2018. [arXiv:1708.00489](https://arxiv.org/abs/1708.00489) | k-Center/Coreset. Metoda porównawcza. |
| 3.11 | ★ | Sinha S., Ebrahimi S., Darrell T., *Variational Adversarial Active Learning*, ICCV, 2019. [doi:10.1109/ICCV.2019.00607](https://doi.org/10.1109/ICCV.2019.00607) | VAAL. Metoda porównawcza. |
| 3.12 | ★ | Munjal P. i in., *Towards Robust and Reproducible Active Learning Using Neural Networks*, CVPR, 2022. [doi:10.1109/CVPR52688.2022.00032](https://doi.org/10.1109/CVPR52688.2022.00032) | Dlaczego random baseline jest tak silny. Bezpośrednio dotyczy I3. |
| 3.13 | ★ | Lüth C.T., Bungert T.J., Klein L., Jaeger P.F., *Navigating the Pitfalls of Active Learning Evaluation: A Systematic Framework for Meaningful Performance Assessment*, NeurIPS, 2023. [arXiv:2301.10625](https://arxiv.org/abs/2301.10625) | Pięć pułapek ewaluacji AL. Checklista do projektu eksperymentów. |

---

## 4. B2 — AL w segmentacji jąder i w histopatologii

Prac stricte o AL dla **instancyjnej segmentacji jąder** jest mało. Ta luka to argument za tematem pracy.
Poniżej najbliższe prace, od najbardziej do najmniej związanych.

| # | | Pozycja | Po co |
|---|---|---|---|
| 4.1 | ★ | Lou W., Li H., Li G., Han X., Wan X., *Which Pixel to Annotate: A Label-Efficient Nuclei Segmentation Framework*, IEEE TMI, 2023. [doi:10.1109/TMI.2022.3221666](https://doi.org/10.1109/TMI.2022.3221666) · [arXiv:2212.10305](https://arxiv.org/abs/2212.10305) · [kod](https://github.com/lhaof/NuSeg) | Wybór łat do adnotacji w segmentacji jąder, ewaluacja m.in. na MoNuSeg. Najbliższa praca. |
| 4.2 | ★ | Qu H., Yi J., Huang Q., Wu P., Metaxas D., *Nuclei Segmentation Using Mixed Points and Masks Selected From Uncertainty*, ISBI, 2020. [doi:10.1109/ISBI45749.2020.9098474](https://doi.org/10.1109/ISBI45749.2020.9098474) | Wybór jąder do pełnej adnotacji na podstawie niepewności (MC). |
| 4.3 | ☆ | Tang i in., *Open-Set Active Learning for Nucleus Detection From the Histopathological Images*, IEEE TMI, 2026. [doi:10.1109/TMI.2025.3617073](https://doi.org/10.1109/TMI.2025.3617073) | Najnowsze AL na jądrach (detekcja, wersja MICCAI'24: OSAL-ND). |
| 4.4 | ☆ | Shen i in., *Deep Active Learning for Breast Cancer Segmentation on Immunohistochemistry Images*, MICCAI, 2020. [doi:10.1007/978-3-030-59722-1_49](https://doi.org/10.1007/978-3-030-59722-1_49) | AL w segmentacji patologii, trzy kryteria wyboru. Porównanie z 3.5 i 3.7. |
| 4.5 | ☆ | Qiu J. i in., *Adaptive Region Selection for Active Learning in Whole Slide Image Semantic Segmentation*, MICCAI, 2023. [arXiv:2307.07168](https://arxiv.org/abs/2307.07168) | Region-based AL na WSI (CAMELYON16). Ważne przy pytaniu o jednostkę zapytania (obraz vs kafel) i przy CAMELYON. |
| 4.6 | ☆ | Lai Z. i in., *Joint Semi-supervised and Active Learning for Segmentation of Gigapixel Pathology Images with Cost-Effective Labeling*, ICCV Workshops, 2021. [doi:10.1109/ICCVW54120.2021.00072](https://doi.org/10.1109/ICCVW54120.2021.00072) | AL + semi-supervised na WSI, koszt adnotacji. |
| 4.7 | ☆ | Li W. i in., *PathAL: An Active Learning Framework for Histopathology Image Analysis*, IEEE TMI, 2022. [doi:10.1109/TMI.2021.3135002](https://doi.org/10.1109/TMI.2021.3135002) | AL w patologii z szumem w etykietach. To klasyfikacja, nie segmentacja, ale problem jest bliski symulowanemu oracle. |

Do dalszego szukania: sekcja „segmentation / pathology” w repo z 3.4 oraz prace cytujące 4.1.

---

## 5. I2 — logika rozmyta typu 1

| # | | Pozycja | Po co |
|---|---|---|---|
| 5.1 | ★ | Zadeh L.A., *Fuzzy sets*, Information and Control, 1965. [doi:10.1016/S0019-9958(65)90241-X](https://doi.org/10.1016/S0019-9958(65)90241-X) | Zbiory rozmyte, funkcje przynależności, operacje. |
| 5.2 | ★ | Mamdani E.H., Assilian S., *An experiment in linguistic synthesis with a fuzzy logic controller*, International Journal of Man-Machine Studies, 1975. [doi:10.1016/S0020-7373(75)80002-2](https://doi.org/10.1016/S0020-7373(75)80002-2) | Wnioskowanie Mamdaniego: reguły, implikacja min, agregacja max, wyostrzanie. Wzór dla I2. |
| 5.3 | ★ | Mendel J.M., *Uncertain Rule-Based Fuzzy Systems: Introduction and New Directions*, 2. wyd., Springer, 2017. [doi:10.1007/978-3-319-51370-6](https://doi.org/10.1007/978-3-319-51370-6) | Rozdziały o systemach typu 1 (rozmycie → reguły → wnioskowanie → wyostrzanie), potem typ 2. Notacja do pracy. |
| 5.4 | ☆ | Mendel J.M., John R.I.B., *Type-2 Fuzzy Sets Made Simple*, IEEE Transactions on Fuzzy Systems, 2002. [doi:10.1109/91.995115](https://doi.org/10.1109/91.995115) | Przejście z typu 1 do IT2 (główny wkład pracy, Etap 2). |

---

## 6. Na później (Faza 4–7 w `docs/PLAN.md`)

- ☆ Tellez D. i in., *Quantifying the effects of data augmentation and stain color normalization in convolutional neural networks for computational pathology*, Medical Image Analysis, 2019. [doi:10.1016/j.media.2019.101544](https://doi.org/10.1016/j.media.2019.101544)
- ☆ Macenko M. i in., *A method for normalizing histology slides for quantitative analysis*, ISBI, 2009. [doi:10.1109/ISBI.2009.5193250](https://doi.org/10.1109/ISBI.2009.5193250)
- ☆ Vahadane A. i in., *Structure-Preserving Color Normalization and Sparse Stain Separation for Histological Images*, IEEE TMI, 2016. [doi:10.1109/TMI.2016.2529665](https://doi.org/10.1109/TMI.2016.2529665)
- ☆ Reinhard E. i in., *Color transfer between images*, IEEE Computer Graphics and Applications, 2001. [doi:10.1109/38.946629](https://doi.org/10.1109/38.946629)
- ☆ Schmidt U., Weigert M., Broaddus C., Myers G., *Cell Detection with Star-Convex Polygons* (StarDist), MICCAI, 2018. [doi:10.1007/978-3-030-00934-2_30](https://doi.org/10.1007/978-3-030-00934-2_30)
- ☆ Hörst F. i in., *CellViT: Vision Transformers for precise cell segmentation and classification*, Medical Image Analysis, 2024. [doi:10.1016/j.media.2024.103143](https://doi.org/10.1016/j.media.2024.103143)
- ☆ Ma J. i in., *Segment anything in medical images* (MedSAM), Nature Communications, 2024. [doi:10.1038/s41467-024-44824-z](https://doi.org/10.1038/s41467-024-44824-z)
