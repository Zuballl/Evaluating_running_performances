# ARCHITEKTURY AUTOENKODERÓW W OCENIE WYDAJNOŚCI SPORTOWEJ: KOMPRESJA DANYCH I RANKING ZAWODNIKÓW

**Mateusz Kubita** Politechnika Warszawska  

**Jan Zubalewicz** Politechnika Warszawska  

**14 marca 2026**

---

### ABSTRAKT
Nagły wzrost dostępności danych z sensorów sportowych (IoT) stwarza wyzwania w zakresie syntezy wielowymiarowych wskaźników wydajności. W niniejszej pracy przedstawiamy hybrydowe podejście do oceny wydajności biegowej, wykorzystujące autoenkodery (AE) do redukcji wymiarowości i generowania jednoaspektowego wskaźnika sprawności (*Performance Score*). Ewaluowaliśmy trzy architektury: liniową (Simple AE), średnio-głęboką (Medium AE) oraz głęboką (Deep AE). Nasz najlepszy model — **Deep Autoencoder** — osiągnął wyjątkowo niski błąd rekonstrukcji (**MSE = 0.000037**), co świadczy o niemal bezstratnej kompresji kluczowych parametrów treningowych do pojedynczej zmiennej ukrytej. Wyniki potwierdzają, że zwiększenie głębokości sieci pozwala na wychwycenie subtelnych, nieliniowych korelacji między biomechaniką a wydolnością organizmu.

---

### 1. Wstęp
W erze cyfryzacji sportu, monitorowanie aktywności fizycznej generuje ogromne zbiory danych zawierające parametry kinetyczne, fizjologiczne i środowiskowe. Kluczowym wyzwaniem pozostaje integracja tych rozproszonych metryk w obiektywny wskaźnik wydajności. Tradycyjne metody często opierają się na arbitralnie dobranych wagach, co może prowadzić do pominięcia złożonych zależności między m.in. mocą, tętnem a prędkością.

W niniejszej pracy eksplorujemy zastosowanie nienadzorowanego uczenia głębokiego do automatycznej ekstrakcji cech wydajnościowych, umożliwiając transparentną i precyzyjną kalibrację rankingu zawodników na podstawie 1.7 miliona rekordów danych.

### 2. Metodologia
Proces badawczy oparto na danych o wysokiej rozdzielczości, poddanych następującym etapom:

1.  **Preprocessing:** Usunięcie rekordów zawierających wartości NaN (665,324 wierszy) oraz normalizacja za pomocą *MinMaxScaler*. Końcowy zbiór danych objął **1,732,810 próbek**.
2.  **Architektury Modeli:**
    * **Simple AE:** Redukcja bezpośrednia (Input -> 1 -> Output).
    * **Medium AE:** Jedna warstwa ukryta (4 neurony) przed wąskim gardłem.
    * **Deep AE:** Struktura wielowarstwowa (5-3-1) zaprojektowana do ekstrakcji hierarchicznych cech.
3.  **Optymalizacja:** Modele trenowano z wykorzystaniem optymalizatora Adam, stosując zwiększony rozmiar paczki (*batch size*) w celu efektywnego przetwarzania dużego zbioru danych.

### 3. Wyniki i Dyskusja
Analiza wykazała, że wraz ze wzrostem złożoności modelu, błąd rekonstrukcji (MSE) ulegał drastycznemu zmniejszeniu. Model głęboki okazał się o dwa rzędy wielkości bardziej precyzyjny od modelu prostego.

| Model Approach | Architecture | MSE | Best For |
| :--- | :--- | :--- | :--- |
| **simple_autoencoder** | Input -> 1 -> Output | 0.004142 | Szybki ranking liniowy |
| **medium_autoencoder** | Input -> 4 -> 1 -> 4 -> Output | 0.000584 | Stabilna kompresja |
| **deep_autoencoder** | Input -> 5 -> 3 -> 1 -> 5 -> 3 -> Output | **0.000037** | **Precyzyjna ocena wydajności** |

Uzyskane wyniki MSE dla modelu **deep_autoencoder** sugerują, że struktura 5-3-1 jest w stanie niemal idealnie zakodować informację o profilu sportowym zawodnika w jednej liczbie (Performance Score). Pozwala to na stworzenie obiektywnego rankingu, który bierze pod uwagę nieliniowe interakcje między parametrami.

### 4. Wnioski
Zastosowanie głębokich autoenkoderów pozwala na obiektywną ocenę wydajności sportowej bez konieczności ręcznego definiowania wag parametrów. Wykazano, że model głęboki najlepiej radzi sobie z syntezą danych sensorycznych, oferując najwyższą wierność odwzorowania rzeczywistego potencjału sportowca. Metoda ta stanowi solidną podstawę do automatycznej identyfikacji liderów oraz outsiderów w dużych zbiorach danych treningowych.
