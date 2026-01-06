# Raport z projektu: ML Running Performance Scorer

## 1. Abstract
Celem projektu jest stworzenie obiektywnego systemu oceny kondycji biegowej, opartego na modelu **uczeniu maszynowym**. Zamiast polegać na prostych statystykach, system analizuje korelację między tempem biegu a kosztem fizjologicznym (tętnem), biorąc pod uwagę parametry biomechaniczne oraz kontekst fizyczny biegacza. 
Dzięki integracji modelu sztucznej inteligencji z czujnikami zawartych w zegarkach Garmin biegacze są w stanie badać swój progres sportowy.

Model został wytrenowany na zbiorze danych obejmującym ponad **170 000 aktywności**, co pozwala na rzetelne odniesienie indywidualnego wyniku do szerokiej populacji biegaczy.

---

## 2. Źródło danych: GoldenCheetah OpenData Project
Kluczowym elementem projektu jest wykorzystanie danych pochodzących z [GoldenCheetah OpenData Project](https://osf.io/6hfpz/overview). Jest to otwarte repozytorium budowane od połowy 2018 roku przez użytkowników popularnej aplikacji do analizy sportowej. W naszym projekcie wykorzystujemy tylko dane biegaczy. 

Dane te zawierają między innymi takie informacje:

| Kolumna | Opis | Jednostka/Format |
| :--- | :--- | :--- |
| **seconds** | Czas trwania od początku aktywności | sekundy |
| **heartrate** | Tętno biegacza zarejestrowane przez pas lub zegarek | bpm (uderzenia/min) |
| **cadence** | Kadencja (liczba kroków na minutę) | spm / rpm |
| **distance** | Całkowity dystans pokonany od startu | metry |
| **altitude** | Wysokość nad poziomem morza | metry |
| **power** | Moc generowana podczas biegu (jeśli użyto czujnika) | Waty (W) |
| **speed** | Prędkość chwilowa | m/s |
| **slope** | Nachylenie terenu (wyliczane z dystansu i wysokości) | procenty (%) |


---

## 3. Wstęp i cel projektu
W naszym projekcie korzystamy algorytmów **sztucznej inteligencji**, które w połączeniu z danymi z **czujników Garmin**, pełnią rolę cyfrowego trenera. System ten został wytrenowany na danych z projektu GoldenCheetah i jest w stanie dostarczyć biegaczowi spersonalizowane rekomendacje treningowe.

Cel tego projektu to dostarczenie narzędzia, które:

* **Weryfikuje realną formę:** Wykorzystuje AI do sprawdzenia wydajności biegowej w oparciu o „surową fizykę” i fizjologię, oddzielając chwilową dyspozycję od faktycznego wzrostu wydolności.
* **Analizuje efektywność:** Ocenia, jak parametry techniczne, takie jak kadencja czy długość kroku, korelują z kosztem energetycznym, wskazując najbardziej efektywny styl biegu.
* **Daje jasny feedback i rekomendacje:** Przekłada skomplikowane dane z sensorów na prostą skalę **0–10** oraz generuje konkretne wskazówki: np. nad jakim parametrem powinieneś popracować.

---

## 4. Model uczenia maszynowego
System wykorzystuje algorytm **PCA (Analiza Składowych Głównych)** do oceny wydajności. Cały proces został podzielony na kilka etapów:

1.  **Przygotowanie danych (`clean_data.py`):** Surowe dane z GoldenCheetah są filtrowane pod kątem błędów sensorów i normalizowane, aby stworzyć spójny zestaw treningowy.
2.  **Trening (`train_model.py`):** Model uczy się rozpoznawać wzorce wydajności, analizując 10 kluczowych cech, m.in. tempo, tętno, nachylenie terenu oraz wiek i wagę biegacza.
3.  **Logika oceny:** Model automatycznie nadaje wagi poszczególnym parametrom. Przykładowo, szybkie tempo przy niskim tętnie i optymalnej kadencji skutkuje wysoką oceną.

---

## 5. Dane z czujnika Garmin
Skrypt `score_my_run.py` pozwala ocenić dowolny trening zapisany w standardowym formacie **.FIT**. Format ten jest dostarczony przez stronę Garmin Connect.

* **Ekstrakcja danych:** Program automatycznie pobiera informacje o tempie, tętnie i dystansie. Potrafi również odczytać dane o użytkowniku bezpośrednio z profilu zapisanego w zegarku.
* **Przetwarzanie sygnału:** Skrypt analizuje m.in. *aerobic decoupling* (dryft tętna), co pozwala ocenić, jak stabilna jest kondycja biegacza w trakcie wysiłku.
* **Raport końcowy:** Użytkownik otrzymuje czytelne zestawienie statystyk oraz ocenę AI wraz z komentarzem „trenera”.

---

## 6. Podsumowanie i dalszy rozwój
W naszym projekcie skorzystaliśmy z publicznie dostępnego repozytorium GoldenCheetah. Wytrenowany przez nas model PCA jest w stanie przedstawić biegaczowi jego estymowany wynik w odniesieniu do szerokiej populacji. 

Co wiecej, **"trener AI"** daje dodatkowe rekomendacje i wskazuje, które wskaźniki odbiegają od normy (np. zbyt wysokie tętno przy małej długości kroku w porównaniu do profesjonalistów). Projekt pozwala na pobranie danych z własnego czujnika Garmina i ich natychmiastową analizę naszym modelem.

**Kolejny krok w rozwoju projektu:**
* **Integracja z Arduino:** Planowane rozszerzenie o autorskie sensory (np. czujniki nacisku w podeszwie do analizy biomechaniki stopy oraz autorskie czujniki tętna), które będą przesyłać dane bezpośrednio do modelu AI.

## 7. Instrukcja 
Wymagany jest **Python 3.8+** oraz następujące biblioteki:

```bash
pip install pandas numpy scikit-learn fitparse joblib seaborn matplotlib goldencheetah-opendata
```

### 2. (Opcjonalne) Wytrenuj model na podstawie danych Cheetah
Jezeli interesuje Cie zbudowanie modelu ucznia maszynwoego od zera

1.  **Pobieranie i obrobka danych:**
    ```bash
    python clean_data.py
    ```
    *Skrypt `fetch_data.py` automatycznie pobierze dane i odpowiednio je przerobi.*

2.  **Wytrenuj model:**
    ```bash
    python train_model.py
    ```
    *Skrypt ten trenuje model PCA i zapisuje go w pliku `performance_scorer.pkl`.*

3. Ocen swoj bieg
    Aby przeanalizować swój trening, użyj skryptu oceniającego. Po wyświetleniu monitu możesz przeciągnąć i upuścić swój plik .FIT bezpośrednio do terminala.

    ```bash
    python score_my_run.py
    ```

---