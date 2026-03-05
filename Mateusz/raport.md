# Raport: Porównanie Architektur Autoenkoderów w Analizie Wydajności Biegowej

## 1. Cel badania
Głównym celem niniejszego projektu była analiza porównawcza różnych architektur **autoenkoderów (AE)** oraz zaawansowanych technik redukcji wymiarowości, takich jak **VAE** (Variational Autoencoder) i **t-SNE**. 

Badanie miało na celu sprawdzenie, w jaki sposób głębokość sieci oraz struktura warstw ukrytych wpływają na zdolność modelu do kompresji danych przy zachowaniu kluczowych informacji o wydajności zawodników. Skupiono się na znalezieniu optymalnego balansu między prostotą modelu a minimalizacją błędu rekonstrukcji (**MSE**).

---

## 2. Zestawienie Wyników Eksperymentalnych

Poniższa tabela podsumowuje wydajność testowanych modeli na podstawie uzyskanych metryk:

| Model | Architektura (Warstwy) | MSE / Metryka | Główne zastosowanie |
| :--- | :--- | :--- | :--- |
| **Simple Autoencoder** | Input -> 1 -> Output | 0.012697 | Modelowanie relacji liniowych |
| **Medium Autoencoder** | Input -> 16 -> 1 -> 16 -> Output | **0.009906** | Ogólne wzorce i nieliniowość |
| **Deep Autoencoder** | Input -> 32 -> 16 -> 8 -> 1 -> ... | 0.009911 | Złożone zależności nieliniowe |
| **VAE** | Probabilistic Latent Space | N/A (KL Div) | Normalizacja rankingów i generowanie |
| **t-SNE** | Manifold Learning | N/A (KL-Loss) | Wizualizacja klastrów i grup |

---

## 3. Analiza Porównawcza i Wnioski

### Wydajność Rekonstrukcji (MSE)
*   **Optymalna złożoność:** Najniższy błąd rekonstrukcji osiągnął **Medium Autoencoder (0.009906)**. Wynik ten sugeruje, że dla badanego zbioru danych umiarkowanie głęboka sieć najlepiej wychwytuje istotne cechy (features) bez wprowadzania nadmiernego szumu.
*   **Problem Simple AE:** Najprostszy model wykazuje najwyższy błąd, co oznacza, że pojedyncza warstwa liniowa nie jest w stanie w pełni odwzorować nieliniowej natury wyników sportowych.
*   **Granica Deep AE:** Model Deep AE uzyskał wynik niemal identyczny z Medium AE, co wskazuje na osiągnięcie "płaskowyżu" uczenia – dodatkowe warstwy nie przyniosły już istotnej poprawy dokładności przy tak silnej kompresji (do 1 wymiaru).

### Przestrzeń Ukryta (Latent Space)
*   Wszystkie autoenkodery zostały skonfigurowane tak, aby ich "wąskie gardło" (bottleneck) wynosiło **1**, co pozwala na redukcję wielu parametrów biegacza do jednej, skalarnej wartości reprezentującej ogólną wydajność (**Performance Score**).

---

## 4. Walidacja Empiryczna: Analiza Skrajnych Przypadków

W celu weryfikacji sensu fizycznego wygenerowanych wyników, przeprowadzono analizę porównawczą dla zawodników o najbardziej skrajnych wartościach `performance_score` (Top 3 vs Bottom 2).

### Tabela 2: Porównanie parametrów rzeczywistych dla skrajnych wyników
| ID Row | Performance Score | Avg Speed [km/h] | Avg Power [W] | Avg Heart Rate | Interpretacja |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **16083** | **9.34966** | 32.11 | 233.11 | 136.34 | **Lider:** Wysoka prędkość przy niskim tętnie. |
| **15741** | **9.29228** | 24.67 | 215.37 | 130.55 | **Top 2:** Wysoka moc i stabilna ekonomia wysiłku. |
| **16271** | **9.26240** | 34.84 | 234.42 | 154.01 | **Top 3:** Ekstremalna prędkość, wyższy koszt biologiczny. |
| **4571** | **-16.0706** | 12.28 | 165.65 | 153.02 | **Bottom 2:** Niska prędkość przy wysokim tętnie. |
| **4575** | **-16.0643** | 11.88 | 165.65 | 153.04 | **Bottom 1:** Najniższa efektywność biegu. |

### Kluczowe obserwacje z walidacji:
1.  **Rozpiętość Wyników:** Model wygenerował wyraźną skalę (od -16 do +9), co pozwala na precyzyjne różnicowanie poziomu zawodników.
2.  **Korelacja z Prędkością:** Zawodnicy z ujemnym wynikiem poruszają się niemal **trzykrotnie wolniej** od liderów rankingu.
3.  **Efektywność (Power-to-HR):** Model poprawnie zinterpretował wysoką wydajność lidera (233W przy 136 bpm) w kontrze do niskiej wydajności dołu tabeli (165W przy 153 bpm).

---

## 5. Podsumowanie i Rekomendacje
Badanie potwierdziło, że wybór architektury ma kluczowe znaczenie dla dokładności modelu. W kontekście oceny wyników biegowych:
*   **Medium Autoencoder** okazał się najbardziej efektywnym narzędziem do precyzyjnego rankingu.
*   **VAE** jest rekomendowany do dalszych prac nad modelami generatywnymi i normalizacją populacji.
*   Model poprawnie mapuje pojęcie "wydajności sportowej" na skalę numeryczną, co ma realne zastosowanie w analityce sportowej.

---
*Autor: mkubita*  
*Data: Marzec 2026*