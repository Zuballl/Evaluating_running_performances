# ARCHITEKTURY AUTOENKODERÓW W OCENIE WYDAJNOŚCI SPORTOWEJ: KOMPRESJA DANYCH I RANKING ZAWODNIKÓW

**Mateusz Kubita**   
Politechnika Warszawska  

**Jan Zubalewicz**  
Politechnika Warszawska  


**5 marca 2026**

---

### ABSTRAKT
Nagły wzrost dostępności danych z sensorów sportowych (IoT) stwarza wyzwania w zakresie syntezy wielowymiarowych wskaźników wydajności. W niniejszej pracy przedstawiamy hybrydowe podejście do oceny wydajności biegowej, wykorzystujące autoenkodery (AE) do redukcji wymiarowości i generowania jednoaspektowego wskaźnika sprawności (*Performance Score*). Ewaluowaliśmy trzy architektury: liniową (Simple AE), średnio-głęboką (Medium AE) oraz głęboką (Deep AE), a także techniki probabilistyczne (VAE) i nieliniowe mapowanie (t-SNE). Nasz najlepszy model — **Medium Autoencoder** — osiągnął najniższy błąd rekonstrukcji (**MSE = 0.009906**), skutecznie kompresując 69 parametrów treningowych do pojedynczej zmiennej ukrytej. Walidacja empiryczna wykazała silną korelację między wygenerowanym wskaźnikiem a fizjologicznymi parametrami sukcesu, takimi jak moc średnia i prędkość, przy jednoczesnym uwzględnieniu ekonomii wysiłku (HR). Wyniki sugerują, że nieliniowe modele kompresji oferują bardziej precyzyjne odzwierciedlenie potencjału sportowego niż tradycyjne średnie ważone.

---

### 1. Wstęp
W erze cyfryzacji sportu, monitorowanie aktywności fizycznej generuje ogromne zbiory danych zawierające parametry kinetyczne, fizjologiczne i środowiskowe (Berduygina et al., 2019). Kluczowym wyzwaniem dla analityków i trenerów pozostaje integracja tych rozproszonych metryk w obiektywny wskaźnik wydajności. Tradycyjne metody często opierają się na arbitralnie dobranych wagach dla poszczególnych cech, co może prowadzić do pominięcia złożonych, nieliniowych zależności między m.in. mocą krytyczną, tętnem a kadencją.

Clickbaitowe lub uproszczone podejście do analizy danych sportowych degraduje jakość informacji i może prowadzić do błędnych decyzji treningowych. W niniejszej pracy eksplorujemy zastosowanie nienadzorowanego uczenia głębokiego do automatycznej ekstrakcji cech wydajnościowych, umożliwiając transparentną i dobrze skalibrowaną predykcję potencjału zawodnika.

### 2. Metodologia
Badania przeprowadzono na zbiorze danych obejmującym parametry aktywności sportowych zintegrowane z danymi o sportowcach. Proces badawczy składał się z następujących etapów:

1.  **Preprocessing:** Wykorzystano rygorystyczny potok czyszczenia danych, obejmujący usuwanie nieskończoności, imputację brakujących wartości medianą oraz usuwanie kolumn o zerowej wariancji. Zastosowano *MinMaxScaler* do normalizacji cech w zakresie [0, 1].
2.  **Architektury Modeli:**
    *   **Simple AE:** Mapowanie bezpośrednie do wymiaru latentnego.
    *   **Medium AE:** Struktura z warstwą ukrytą o rozmiarze 16 neuronów.
    *   **Deep AE:** Architektura wielowarstwowa (32-16-8) dla uchwycenia hierarchicznych zależności.
3.  **Metryka sukcesu:** Głównym kryterium oceny był błąd średniokwadratowy (*Mean Squared Error*, MSE) rekonstrukcji danych wejściowych z jednowymiarowej przestrzeni ukrytej.

### 3. Wyniki i Dyskusja
Analiza błędów rekonstrukcji wykazała, że architektura Medium AE oferuje optymalny balans (MSE = 0.009906). Model prosty (Simple AE) nie był w stanie odwzorować nieliniowości danych, natomiast model głęboki (Deep AE) nie przyniósł znaczącej poprawy, co wskazuje na osiągnięcie "płaskowyżu" informacyjnego.

| Model | Architektura (Warstwy) | MSE | Główne zastosowanie |
| :--- | :--- | :--- | :--- |
| **Simple AE** | Input -> 1 -> Output | 0.012697 | Modelowanie liniowe |
| **Medium AE** | Input -> 16 -> 1 -> 16 -> Output | **0.009906** | **Optymalny ranking** |
| **Deep AE** | Input -> 32 -> 16 -> 8 -> 1 -> ... | 0.009911 | Złożone zależności |

### 4. Wnioski
Zastosowanie autoenkoderów pozwala na obiektywną ocenę wydajności sportowej bez konieczności ręcznego definiowania wag dla dziesiątek parametrów. Wykazano, że średnio-głębokie sieci neuronowe są najskuteczniejsze w syntezie danych sensorycznych, oferując stabilną podstawę do rankingu zawodników. Proponowany zestaw cech zwiększa interpretowalność wyników poprzez podkreślenie istotnych powiązań między mechaniką ruchu a kosztem biologicznym.

---
*Kod źródłowy oraz wyuczone modele zostały udostępnione w celu wsparcia replikowalności badań.*