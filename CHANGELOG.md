# Changelog

## 0.3.1 - Dashboard i overlay LLM

### Dodano
- Karta `Model LLM` w dashboardzie dyktowania z informacją o backendzie, modelu i stanie połączenia.
- Automatyczne sprawdzanie połączenia z lokalnym LLM po starcie aplikacji.
- Ręczne sprawdzanie połączenia LLM z poziomu widoku `Modele`.
- Widoczny stan, czy LLM będzie przetwarzał kolejne nagrania, czy jest dostępny tylko ręcznie.

### Zmieniono
- Dashboard lepiej układa cztery główne karty: `Status`, `Model`, `Auto-paste`, `Model LLM`.
- Główne okno aplikacji jest szersze, żeby nowe elementy dashboardu nie były ucinane.
- Pływający overlay pokazuje etapy przetwarzania po zatrzymaniu nagrania.
- Podczas transkrypcji i pracy LLM ta sama główna falka zmienia tryb animacji zamiast dodawać osobny pasek.
- Gdy LLM jest włączony dla nagrania, auto-paste wkleja wygenerowany prompt zamiast surowej transkrypcji.

## 0.3.0 - Lokalny LLM do porządkowania transkrypcji

### Dodano
- Drugi etap po Whisperze: lokalny LLM może przerobić surową transkrypcję na uporządkowany prompt.
- Obsługa backendu Ollama.
- Obsługa backendu `llama-cpp-python` z lokalnym plikiem `.gguf`.
- Presety modeli LLM, w tym `qwen3:1.7b` jako domyślna rekomendacja.
- Przechowywanie w historii obu wersji tekstu: transkrypcji oraz promptu.
- Przełączane podglądy `Transkrypcja` i `Prompt`.
- Akcje kopiowania transkrypcji, kopiowania promptu i ponownego przetwarzania.
- Przełącznik LLM dla kolejnych nagrań oraz przełącznik `LLM ON/OFF` w overlayu nagrywania.

### Zmieniono
- Konfiguracja lokalnego LLM znajduje się w widoku `Modele`.
- Historia pokazuje status przetwarzania LLM i ewentualne błędy.
- Przy błędzie LLM aplikacja zachowuje oryginalną transkrypcję i może wkleić ją jako fallback.

## 0.2.0 - Wygodniejsze dyktowanie

### Dodano
- Dashboard dyktowania ze statusem, aktualnym modelem i stanem auto-paste.
- Pływający widget nagrywania z timerem, limitem czasu i falą głosu.
- Globalny skrót do startu i zatrzymania nagrywania.
- Automatyczne wklejanie tekstu do aktywnego okna.
- Historia transkrypcji z podglądem i kopiowaniem wpisów.
- Widok diagnostyki dla środowiska, schowka i logów.

### Zmieniono
- Rozbudowano ustawienia nagrywania, transkrypcji, schowka, hotkeyów i wyglądu.
- Poprawiono obsługę X11 i Wayland dla schowka oraz auto-paste.

## 0.1.0 - Pierwsza wersja

### Dodano
- Aplikacja GUI do lokalnej transkrypcji mowy na tekst.
- Obsługa modeli Whisper.
- Wybór modelu transkrypcji i języka.
- Zapis nagrań audio oraz wyników transkrypcji.
