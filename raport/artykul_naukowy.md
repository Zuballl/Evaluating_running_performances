# ARCHITEKTURY AUTOENKODERÓW W OCENIE WYDAJNOŚCI SPORTOWEJ: KOMPRESJA DANYCH I RANKING ZAWODNIKÓW

**Mateusz Kubita** Politechnika Warszawska  
**Jan Zubalewicz** Politechnika Warszawska  
**14 marca 2026**

---

## ABSTRAKT

Nagły wzrost dostępności danych z sensorów sportowych (IoT) stwarza wyzwania w zakresie syntezy wielowymiarowych wskaźników wydajności. W niniejszej pracy przedstawiamy hybrydowe podejście do oceny wydajności biegowej, wykorzystujące nienadzorowane metody uczenia maszynowego (w tym autoenkodery i modele probabilistyczne) do redukcji wymiarowości i generowania jednoaspektowego wskaźnika sprawności (*Performance Score*). Ewaluowaliśmy pięć różnych architektur. Nasza analiza błędu rekonstrukcji (MSE) ujawniła, że nieliniowe metody, w szczególności **Variational Autoencoder (VAE)** (MSE = **0.002813**) i **Deep Autoencoder** (MSE = **0.003822**), osiągają doskonałe wyniki, znacznie przewyższając prosty model Simple AE. Choć model liniowy (PCA) wykazał najniższy błąd (**MSE = 0.001150**), potwierdzając silny komponent liniowy danych, hierarchiczna kompresja nieliniowa oferuje głębszy wgląd w subtelne interakcje między biomechaniką a wydolnością organizmu, umożliwiając przejrzystą interpretację tego, co definiuje liderów i outsiderów.

---

## 1. Wstęp i Cel Projektu

W erze cyfryzacji sportu, monitorowanie aktywności fizycznej generuje ogromne zbiory danych zawierające parametry kinetyczne, fizjologiczne i środowiskowe (np. tętno, tempo, kadencja, przewyższenia). Tradycyjne metody oceny często opierają się na arbitralnie dobranych wagach, co może prowadzić do pominięcia złożonych zależności.

**Głównym celem tego projektu** jest stworzenie w pełni obiektywnego, opartego na danych (data-driven) systemu oceny wydajności. Zamiast ręcznie definiować, co oznacza "dobry trening", wykorzystujemy metody redukcji wymiarowości do skompresowania wielowymiarowego profilu zawodnika do jednej wartości (tzw. wąskie gardło autoenkodera). Ta pojedyncza wartość (*Performance Score*) pozwala nie tylko na stworzenie sprawiedliwego rankingu sportowców, ale również, dzięki zastosowaniu technik interpretowalności (XAI), na zidentyfikowanie parametrów, które najbardziej decydują o sukcesie.

---

## 2. Metodologia

### 2.1. Zbiór danych i Preprocessing

Zbiór danych pochodzi z projektu **GoldenCheetah OpenData**, inicjatywy mającej na celu udostępnienie zasobów danych treningowych dla celów badawczych z zachowaniem prywatności użytkowników. W momencie publikacji obejmował on ponad 1300 unikalnych sportowców i przeszło 700 000 zarejestrowanych aktywności. Ponieważ surowe dane crowdsourcingowe cechują się dużą wariancją jakości, wdrożono rygorystyczny proces przygotowania danych (preprocessing):

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

Sam *Performance Score* nie tłumaczy, *dlaczego* dany zawodnik jest oceniany wyżej. Aby zidentyfikowanie najważniejsze parametry wpływające na ten wynik, opracowaliśmy autorską, syntetyczną metrykę **Combined Impact**. Wyniki z czterech różnych metod są normalizowane do zakresu od 0 do 1, a następnie uśredniane:

$Combined\_Impact = \frac{Spearman_{norm} + Kendall_{norm} + MI_{norm} + Permutation_{norm}}{4}$

Dzięki temu ranking cech łączy korelacje liniowe, odporność na wartości odstające (Kendall), nieliniowe relacje informacyjne (Mutual Information) oraz weryfikację predykcyjną na modelu Random Forest (Permutation Importance).

![Najważniejsze cechy per model](top_features_per_model.png)
*Rysunek 1: Znaczenie poszczególnych zmiennych (TOP features) obliczone na podstawie Combined Impact dla ewaluowanych modeli.*

---

## 3. Wyniki i Dyskusja

### 3.1. Błąd Rekonstrukcji (MSE) - Aktualne Wyniki
Analiza błędu rekonstrukcji (MSE) ujawniła zróżnicowane wyniki. Model liniowy (PCA) osiągnął najniższy błąd, co sugeruje, że znaczna część wariancji w danych jest liniowa. Spośród modeli nieliniowych, VAE i Deep Autoencoder osiągnęły najlepsze wyniki, znacznie przewyższając prosty model Simple AE.

| Model Approach | Architecture | MSE (Aktualne) | Best For |
| :--- | :--- | :--- | :--- |
| **pca** | Input -> PCA(1) -> Output | **0.001150** | Szybki baseline liniowy, niska utrata wariancji |
| **simple_autoencoder** | Input -> 1 -> Output | **0.051031** | Prosty ranking nieliniowy (najgorsza rekonstrukcja) |
| **medium_autoencoder** | Input -> 4 -> 1 -> 4 -> Output | **0.005125** | Stabilna kompresja nieliniowa |
| **vae** | Input -> Dense -> z_mean(1) -> Output | **0.002813** | Probabilistyczne modelowanie przestrzeni |
| **deep_autoencoder** | Input -> 5 -> 3 -> 1 -> 3 -> 5 -> Output | **0.003822** | Precyzyjna kompresja hierarchiczna nieliniowa |

Uzyskane wyniki MSE sugerują, że nieliniowe zależności nie dominują całkowicie w tym zbiorze danych, o czym świadczy doskonały wynik PCA. Jednakże, doskonała rekonstrukcja osiągnięta przez VAE i Deep Autoencoder (odpowiednio **0.002813** i **0.003822**) wskazuje, że te modele są w stanie skompresować profil sportowy zawodnika do jednej wartości (Performance Score) z minimalną utratą informacji nieliniowych. Pozwala to na stworzenie obiektywnego rankingu, który bierze pod uwagę złożone interakcje.

![Porównanie błędu MSE](mse_comparison.png)
*Rysunek 2: Zestawienie błędu średniokwadratowego (MSE) dla wszystkich wytrenowanych architektur (według aktualnych wyników).*

### 3.2. Zgodność Modeli (Model Agreement & Latent Scores)
Eksploracja przestrzeni ukrytej ujawniła zróżnicowane podejścia modeli do kompresji wskaźników wydajnościowych. 

![Zgodność wyników między modelami](model_agreement.png)
*Rysunek 3: Macierz zgodności poszczególnych architektur w ocenie wydajności.*

![Porównanie ukrytych wyników (Latent Score)](latent_score_comparison.png)
*Rysunek 4: Dystrybucja wyników Latent Score między różnymi podejściami do modelowania (PCA vs Autoenkodery vs VAE).*


---

## 4. Wnioski

Zastosowanie głębokich autoenkoderów i modeli probabilistycznych pozwala na rzetelną ocenę wydajności sportowców bez konieczności ręcznego definiowania wag parametrów. Wykazano, że nieliniowe metody, w szczególności Variational Autoencoder (VAE) (MSE=0.002813) i Deep Autoencoder (MSE=0.003822), skutecznie radzą sobie z syntezą danych sensorycznych przy zachowaniu wysokiej wierności (niski MSE). Choć model liniowy (PCA) osiągnął najniższy błąd rekonstrukcji, potwierdzając silny komponent liniowy danych, nieliniowe podejście AE/VAE oferuje wyższą precyzję w modelowaniu złożonych relacji w przestrzeni ukrytej. Ponadto, wprowadzenie autorskiej miary *Combined Impact* rozwiązuje problem "czarnej skrzynki" (black-box) typowy dla sieci neuronowych, dostarczając trenerom i analitykom czytelnej informacji o kluczowych czynnikach warunkujących sukces sportowy. System ten stanowi kompletną platformę do automatycznej identyfikacji talentów oraz optymalizacji analizy treningowej.