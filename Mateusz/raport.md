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
| **VAE** (Zadanie poprzednie) | Probabilistic Latent Space | N/A (KL Div) | Normalizacja rankingów i generowanie |
| **t-SNE** (Zadanie poprzednie) | Manifold Learning | N/A (KL-Loss) | Wizualizacja klastrów i grup |

---

## 3. Analiza Porównawcza i Wnioski

### Wydajność Rekonstrukcji (MSE)
*   **Optymalna złożoność:** Najniższy błąd rekonstrukcji osiągnął **Medium Autoencoder (0.009906)**. Wynik ten sugeruje, że dla badanego zbioru danych umiarkowanie głęboka sieć najlepiej wychwytuje istotne cechy (features) bez wprowadzania nadmiernego szumu.
*   **Problem Simple AE:** Najprostszy model (Simple AE) wykazuje najwyższy błąd, co oznacza, że pojedyncza warstwa liniowa nie jest w stanie w pełni odwzorować nieliniowej natury wyników sportowych.
*   **Granica Deep AE:** Model Deep AE uzyskał wynik niemal identyczny z Medium AE, co wskazuje na osiągnięcie "płaskowyżu" uczenia – dodatkowe warstwy nie przyniosły już istotnej poprawy dokładności przy tak silnej kompresji (do 1 wymiaru).

### Przestrzeń Ukryta (Latent Space)
*   Wszystkie autoenkodery zostały skonfigurowane tak, aby ich "wąskie gardło" (bottleneck) wynosiło **1**, co pozwala na redukcję wielu parametrów biegacza do jednej, skalarnej wartości reprezentującej ogólną wydajność.
*   W przeciwieństwie do standardowych AE, **VAE** pozwala na uzyskanie ciągłej przestrzeni parametrów, co jest bardziej użyteczne przy statystycznej ocenie populacji biegaczy, mimo że nie zawsze optymalizuje czysty błąd MSE.

## 4. Podsumowanie
Badanie potwierdziło, że wybór architektury ma kluczowe znaczenie dla dokładności modelu. W kontekście oceny wyników biegowych, **Medium Autoencoder** okazał się najbardziej efektywnym narzędziem, oferując precyzyjną kompresję danych przy zachowaniu wysokiej zdolności do ich późniejszej rekonstrukcji.

---
*Autor: mkubita*
*Data: Marzec 2026*