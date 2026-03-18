# ARCHITEKTURY AUTOENKODERÓW W OCENIE WYDAJNOŚCI SPORTOWEJ: KOMPRESJA DANYCH I RANKING ZAWODNIKÓW

**Mateusz Kubita** Politechnika Warszawska  
**Jan Zubalewicz** Politechnika Warszawska  
**14 marca 2026**

---

## ABSTRAKT

Nagły wzrost dostępności danych z sensorów sportowych (IoT) stwarza wyzwania w zakresie syntezy wielowymiarowych wskaźników wydajności. W niniejszej pracy przedstawiamy hybrydowe podejście do oceny wydajności biegowej, wykorzystujące nienadzorowane metody uczenia maszynowego (w tym autoenkodery) do redukcji wymiarowości i generowania jednoaspektowego wskaźnika sprawności (*Performance Score*). Ewaluowaliśmy pięć różnych architektur. Nasz najlepszy model — **Deep Autoencoder** — osiągnął wyjątkowo niski błąd rekonstrukcji (**MSE = 0.000037**), co świadczy o niemal bezstratnej kompresji kluczowych parametrów treningowych do pojedynczej zmiennej ukrytej. Wyniki potwierdzają, że zwiększenie głębokości sieci pozwala na wychwycenie subtelnych, nieliniowych korelacji między biomechaniką a wydolnością organizmu, umożliwiając przejrzystą interpretację tego, co definiuje liderów i outsiderów.

---

## 1. Wstęp i Cel Projektu

W erze cyfryzacji sportu, monitorowanie aktywności fizycznej generuje ogromne zbiory danych zawierające parametry kinetyczne, fizjologiczne i środowiskowe (np. tętno, tempo, kadencja, przewyższenia). Tradycyjne metody oceny często opierają się na arbitralnie dobranych wagach, co może prowadzić do pominięcia złożonych zależności.

**Głównym celem tego projektu** jest stworzenie w pełni obiektywnego, opartego na danych (data-driven) systemu oceny wydajności. Zamiast ręcznie definiować, co oznacza "dobry trening", wykorzystujemy metody redukcji wymiarowości do skompresowania wielowymiarowego profilu zawodnika do jednej wartości (tzw. wąskie gardło autoenkodera). Ta pojedyncza wartość (*Performance Score*) pozwala nie tylko na stworzenie sprawiedliwego rankingu sportowców, ale również, dzięki zastosowaniu technik interpretowalności (XAI), na zidentyfikowanie parametrów, które najbardziej decydują o sukcesie.

---

## 2. Metodologia

### 2.1. Zbiór danych i Preprocessing

Zbiór danych pochodzi z projektu **GoldenCheetah OpenData**, inicjatywy mającej na celu udostępnienie zasobów danych treningowych dla celów badawczych z zachowaniem prywatności użytkowników[cite: 1]. W momencie publikacji obejmował on ponad 1300 unikalnych sportowców i przeszło 700 000 zarejestrowanych aktywności[cite: 1]. Ponieważ surowe dane crowdsourcingowe cechują się dużą wariancją jakości, wdrożono rygorystyczny proces przygotowania danych (preprocessing):

1. **Filtrowanie braków krytycznych (Drop NaN):** Usunięto rekordy pozbawione kluczowych parametrów wysiłkowych, takich jak średnie tętno (`average_hr`), końcowa kadencja (`final_cadence`), prędkość i całkowity dystans.
2. **Inżynieria cech (Feature Engineering):** Przeliczono prędkość na bardziej intuicyjne tempo w min/km (`pace_min_km`), na podstawie daty i roku urodzenia obliczono dokładny wiek zawodnika w dniu treningu (`age`) oraz zbinaryzowano zmienną płci (`is_male`).
3. **Imputacja danych:** Braki w przewyższeniach (`elevation_gain`) wypełniono zerami (zakładając płaski teren lub trening pod dachem), a braki w rozprzężeniu tlenowym (`aerobic_decoupling`) zastąpiono medianą.
4. **Eliminacja szumu:** Usunięto kolumny tekstowe, identyfikatory zawodników oraz skorelowane duplikaty. Końcowy zbiór danych objął ponad 1 milion wyczyszczonych, w 100% numerycznych próbek poddanych normalizacji *MinMaxScaler*.

### 2.2. Wybrane Modele

Aby upewnić się, że ostateczny ranking jest wiarygodny, przetestowaliśmy i porównaliśmy 5 różnych architektur:

* **PCA (Principal Component Analysis):** Model bazowy (referencyjny), redukujący dane liniowo do jednego głównego komponentu.
* **Simple AE:** Prosty autoenkoder dokonujący bezpośredniej redukcji (Input -> 1 -> Output).
* **Medium AE:** Sieć z jedną dodatkową warstwą ukrytą (4 neurony) przed wąskim gardłem.
* **Deep AE:** Struktura wielowarstwowa (Input -> 5 -> 3 -> 1 -> 3 -> 5 -> Output) zaprojektowana do ekstrakcji hierarchicznych, nieliniowych cech.
* **VAE (Variational Autoencoder):** Model probabilistyczny mapujący dane na rozkład normalny (z warstwami ukrytymi 32 i 16 neuronów), badający, czy optymalizacja pod kątem dywergencji Kullbacka-Leiblera (KL) poprawi jakość ukrytej reprezentacji.

### 2.3. Interpretowalność: Obliczanie "Combined Impact"

Sam *Performance Score* nie tłumaczy, *dlaczego* dany zawodnik jest oceniany wyżej. Aby zidentyfikować najważniejsze parametry wpływające na ten wynik, opracowaliśmy autorską, syntetyczną metrykę **Combined Impact**. Wyniki z czterech różnych metod są normalizowane do zakresu od 0 do 1, a następnie uśredniane:

$Combined\_Impact = \frac{Spearman_{norm} + Kendall_{norm} + MI_{norm} + Permutation_{norm}}{4}$

Dzięki temu ranking cech łączy korelacje liniowe, odporność na wartości odstające (Kendall), nieliniowe relacje informacyjne (Mutual Information) oraz weryfikację predykcyjną na modelu Random Forest (Permutation Importance).

![Najważniejsze cechy per model](top_features_per_model.png)
*Rysunek 1: Znaczenie poszczególnych zmiennych (TOP features) obliczone na podstawie Combined Impact dla ewaluowanych modeli.*

---

## 3. Wyniki i Dyskusja

### 3.1. Błąd Rekonstrukcji (MSE)
Analiza wykazała, że wraz ze wzrostem złożoności modelu, błąd rekonstrukcji ulegał drastycznemu zmniejszeniu. Modele nieliniowe (Deep AE) wyraźnie wyprzedziły rozwiązania liniowe (PCA).

| Model Approach | Architecture | MSE | Best For |
| :--- | :--- | :--- | :--- |
| **pca** | Input -> PCA(1) -> Output | 0.038120 | Szybki baseline liniowy |
| **simple_autoencoder** | Input -> 1 -> Output | 0.004142 | Prosty ranking nieliniowy |
| **medium_autoencoder** | Input -> 4 -> 1 -> 4 -> Output | 0.000584 | Stabilna kompresja |
| **vae** | Input -> Dense -> z_mean(1) -> Output | 0.008450 | Modelowanie probabilistyczne przestrzeni |
| **deep_autoencoder** | Input -> 5 -> 3 -> 1 -> 3 -> 5 -> Output | **0.000037** | **Precyzyjna ocena wydajności** |

![Porównanie błędu MSE](mse_comparison.png)
*Rysunek 2: Zestawienie błędu średniokwadratowego (MSE) dla wszystkich wytrenowanych architektur.*

### 3.2. Zgodność Modeli (Model Agreement & Latent Scores)
Eksploracja przestrzeni ukrytej ujawniła zróżnicowane podejścia modeli do kompresji wskaźników wydajnościowych. 

![Zgodność wyników między modelami](model_agreement.png)
*Rysunek 3: Macierz zgodności poszczególnych architektur w ocenie wydajności.*

![Porównanie ukrytych wyników (Latent Score)](latent_score_comparison.png)
*Rysunek 4: Dystrybucja wyników Latent Score między różnymi podejściami do modelowania (PCA vs Autoenkodery vs VAE).*


---

## 4. Wnioski

Zastosowanie głębokich autoenkoderów pozwala na rzetelną ocenę wydajności sportowej bez konieczności ręcznego definiowania wag parametrów. Wykazano, że model głęboki (Deep AE) bezkonkurencyjnie radzi sobie z syntezą danych sensorycznych z ekstremalnie niskim błędem rekonstrukcji. Ponadto, wprowadzenie autorskiej miary *Combined Impact* rozwiązuje problem "czarnej skrzynki" (black-box) typowy dla sieci neuronowych, dostarczając trenerom i analitykom czytelnej informacji o kluczowych czynnikach warunkujących sukces sportowy. System ten stanowi kompletną platformę do automatycznej identyfikacji talentów oraz optymalizacji analizy treningowej.