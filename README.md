# Azor Transcriber

Azor Transcriber to desktopowa aplikacja do szybkiego dyktowania tekstu po polsku i automatycznego wklejania transkrypcji do aktywnego pola tekstowego. Aplikacja nagrywa głos z mikrofonu, przepisuje go lokalnie przez modele Whisper z Hugging Face, zapisuje historię transkrypcji i może od razu wkleić wynik tam, gdzie aktualnie piszesz.

Projekt jest szczególnie przydatny, gdy często piszesz dłuższe wiadomości, komentarze w kodzie, notatki, opisy zadań albo odpowiedzi w komunikatorach i chcesz mówić zamiast pisać. Najwygodniejszy tryb pracy to ustawienie kursora w dowolnym inpucie, użycie globalnego skrótu, podyktowanie tekstu i pozwolenie aplikacji, żeby po transkrypcji sama wkleiła wynik.

Główna aplikacja znajduje się w pliku `app.py`.

## Screenshoty

Po dodaniu screenshotów do repozytorium możesz podmienić poniższe ścieżki na właściwe pliki.

### Główny widok aplikacji

![Główny widok Azor Transcriber](docs/screenshots/main-dashboard.png)

### Overlay nagrywania

![Overlay nagrywania Azor Transcriber](docs/screenshots/recording-overlay.png)

## Funkcje

- Nagrywanie audio z wybranego mikrofonu.
- Transkrypcja mowy przez modele Whisper uruchamiane przez `transformers`.
- Obsługa modeli ogólnych i modeli dostrojonych pod język polski.
- Automatyczne kopiowanie wyniku do schowka.
- Automatyczne wklejanie transkrypcji do aktywnego pola tekstowego.
- Globalny skrót do startu i zatrzymania dyktowania.
- Historia transkrypcji z możliwością kopiowania i usuwania wpisów.
- Widok modeli z informacją o jakości, rozmiarze i rekomendowanym użyciu.
- Widok ustawień dla mikrofonu, języka, limitu nagrywania, hotkeya i auto-paste.
- Diagnostyka środowiska, schowka, narzędzi systemowych i logów.
- Overlay nagrywania z licznikiem, falą głosu i przyciskiem stop.

## Wymagania

- Python 3.10 lub nowszy.
- Mikrofon działający w systemie.
- Tkinter.
- PortAudio wymagane przez `PyAudio`.
- Połączenie z internetem przy pierwszym pobraniu modelu z Hugging Face.
- Miejsce na dysku na modele Whisper i zależności Pythona.

CUDA nie jest wymagana, ale mocno przyspiesza transkrypcję. Bez GPU aplikacja działa na CPU, tylko większe modele mogą być zbyt wolne do wygodnego dyktowania.

## Instalacja Zależności Systemowych

### Linux / Ubuntu / Debian

```bash
sudo apt-get update
sudo apt-get install portaudio19-dev python3-tk
```

Dla najlepszego działania schowka i auto-paste na X11:

```bash
sudo apt-get install xdotool xclip
```

Dla Waylanda:

```bash
sudo apt-get install wl-clipboard
```

Opcjonalnie:

```bash
sudo apt-get install ffmpeg libnotify-bin
```

`libnotify-bin` dostarcza `notify-send`, którego aplikacja używa do prostych powiadomień desktopowych, jeśli narzędzie jest dostępne.

### macOS

```bash
brew install portaudio
brew install tcl-tk
```

Na macOS auto-paste i globalne skróty mogą wymagać dodatkowych uprawnień systemowych w ustawieniach prywatności i dostępności.

## Instalacja Projektu

Wejdź do katalogu projektu:

```bash
cd transcriber-ui
```

Utwórz i aktywuj środowisko wirtualne:

```bash
python -m venv .venv
source .venv/bin/activate
```

Na Windows aktywacja wygląda inaczej:

```bash
.venv\Scripts\activate
```

Zainstaluj zależności:

```bash
pip install -r requirements.txt
```

Jeśli chcesz używać CUDA, sprawdź aktualną instrukcję instalacji `torch` dla swojej wersji sterowników i CUDA. W praktyce może być potrzebna osobna instalacja `torch` z właściwego indexu PyTorch przed albo po instalacji `requirements.txt`.

## Uruchomienie

```bash
source .venv/bin/activate
python app.py
```

Przy pierwszym uruchomieniu aplikacja:

- utworzy katalog `output/`, jeśli go nie ma,
- zapisze domyślne ustawienia w `output/settings.json`,
- zacznie ładować domyślny model `openai/whisper-large-v3-turbo`,
- może pobrać model do cache Hugging Face,
- utworzy log w pliku `transcriber.log`.

Pierwsze ładowanie modelu może potrwać dłużej, bo model musi zostać pobrany i zapisany lokalnie.

## Pierwsza Konfiguracja

Po uruchomieniu warto przejść przez kilka ustawień:

1. Otwórz zakładkę `Modele`.
2. Wybierz model odpowiedni do swojego sprzętu.
3. Kliknij `Załaduj model`, jeśli wybrany model nie jest jeszcze aktywny.
4. Otwórz zakładkę `Ustawienia`.
5. Wybierz mikrofon.
6. Ustaw język transkrypcji: `polish`, `english` albo `auto`.
7. Ustaw limit nagrywania i zachowanie auto-paste.
8. Zapisz ustawienia.

Rekomendacje modeli:

- CPU: `openai/whisper-small`
- GPU: `openai/whisper-large-v3-turbo`
- lepsza jakość dla polskiego: `bardsai/whisper-medium-pl`
- szybki test: `openai/whisper-tiny`

Większe modele zwykle dają lepszą jakość, ale wymagają więcej pamięci i czasu. Na CPU warto zacząć od mniejszych modeli.

## Jak Korzystać

Najprostszy przepływ pracy:

1. Uruchom aplikację.
2. Poczekaj, aż model się załaduje.
3. Ustaw kursor w miejscu, w którym chcesz wkleić tekst.
4. Użyj globalnego skrótu albo kliknij `Start dyktowania`.
5. Powiedz tekst do mikrofonu.
6. Zatrzymaj nagrywanie albo poczekaj na automatyczny limit.
7. Aplikacja zapisze audio, wykona transkrypcję i pokaże wynik.
8. Jeśli auto-paste jest włączony, tekst zostanie wklejony do aktywnego inputu.

Domyślny globalny skrót:

```text
<ctrl>+<alt>+n
```

Jeśli auto-paste się nie uda, aplikacja próbuje zostawić wynik w schowku. Możesz wtedy wkleić tekst ręcznie przez `Ctrl+V`.

## Widoki Aplikacji

### Dyktowanie

Główny widok pracy. Pozwala rozpocząć i zatrzymać nagrywanie, zobaczyć status modelu, sprawdzić ostatnią transkrypcję, skopiować tekst i wykonać test auto-paste.

### Historia

Zawiera poprzednie transkrypcje. Możesz wybrać wpis, skopiować go do schowka, usunąć pojedynczą transkrypcję albo wyczyścić całą historię.

### Modele

Pozwala wybrać model Whisper i sprawdzić jego rozmiar, jakość, szybkość oraz rekomendowane użycie. Aplikacja pokazuje też, czy działa na CPU czy GPU.

### Ustawienia

Zawiera konfigurację nagrywania, transkrypcji, schowka, auto-paste, hotkeya, wyglądu i zachowania na X11/Wayland.

### Diagnostyka

Pomaga sprawdzić środowisko, urządzenia audio, dostępność narzędzi `xdotool`, `xclip`, `wl-copy`, działanie schowka i ostatnie linie logów.

## Auto-paste, X11 i Wayland

Auto-paste działa najlepiej na X11, gdy zainstalowane są:

```bash
sudo apt-get install xdotool xclip
```

Na X11 aplikacja może:

- zapamiętać aktywne okno przed nagrywaniem,
- skopiować tekst jako `text/plain`,
- przywrócić fokus do poprzedniego okna,
- wysłać `Ctrl+V`.

Na Waylandzie system może blokować globalne skróty, aktywację okien i automatyczne wysyłanie klawiszy. W takim przypadku aplikacja nadal może skopiować wynik do schowka, ale ręczne `Ctrl+V` może być bardziej niezawodne.

Jeśli zależy Ci na najbardziej przewidywalnym auto-paste, używaj sesji X11.

## Pliki Tworzone Przez Aplikację

```text
output/settings.json
output/transcription-history.json
output/recording-*.wav
transcriber.log
```

Znaczenie plików:

- `output/settings.json` - zapisane ustawienia aplikacji.
- `output/transcription-history.json` - historia transkrypcji.
- `output/recording-*.wav` - nagrania audio używane do transkrypcji.
- `transcriber.log` - log działania aplikacji.

Katalog `output/` może rosnąć, szczególnie jeśli często nagrywasz dłuższe fragmenty. Warto okresowo usuwać niepotrzebne pliki `.wav`.

Modele pobierane przez Hugging Face są zwykle przechowywane poza projektem, w cache użytkownika, na przykład w `~/.cache/huggingface/`.

## Budowanie Aplikacji Desktopowej

Projekt ma zależność `pyinstaller`, więc można zbudować prostą wersję desktopową:

```bash
pyinstaller --onefile --windowed --name "Azor-Transcriber" app.py
```

Budowanie może potrwać dłużej, a wynikowy plik może być duży przez `torch`, `transformers`, `PyAudio` i pozostałe zależności.

## Rozwiązywanie Problemów

### Aplikacja nie widzi mikrofonu

Sprawdź, czy mikrofon działa w systemie i czy `PyAudio` poprawnie się zainstalował. Na Linuxie upewnij się, że masz `portaudio19-dev`.

### Brakuje Tkintera

Na Linuxie doinstaluj:

```bash
sudo apt-get install python3-tk
```

Na macOS:

```bash
brew install tcl-tk
```

### Model ładuje się bardzo długo

Pierwsze ładowanie może pobierać model z internetu. Większe modele mogą zajmować od kilkuset MB do kilku GB.

### Transkrypcja jest wolna

Sprawdź w zakładce `Modele` albo `Diagnostyka`, czy aplikacja używa CUDA. Jeśli działa na CPU, wybierz mniejszy model, na przykład `openai/whisper-small`.

### Auto-paste nie działa

Sprawdź zakładkę `Diagnostyka`. Na X11 zainstaluj `xdotool` i `xclip`. Na Waylandzie auto-paste może być ograniczony przez system, wtedy użyj ręcznego `Ctrl+V`.

### Globalny skrót nie działa

Sprawdź, czy zależność `pynput` jest zainstalowana i czy środowisko graficzne pozwala na globalne skróty. Na Waylandzie może to działać niestabilnie.

### Gdzie szukać błędów

Otwórz zakładkę `Diagnostyka` i odśwież logi albo sprawdź plik:

```text
transcriber.log
```

## Struktura Projektu

```text
app.py
clipboard_manager.py
paste_manager.py
tkinter-only.py
requirements.txt
output/
```

Opis:

- `app.py` - główna aplikacja GUI.
- `clipboard_manager.py` - kopiowanie, odczyt i weryfikacja schowka.
- `paste_manager.py` - auto-paste, przywracanie okna i wysyłanie skrótu `Ctrl+V`.
- `tkinter-only.py` - prosty, historyczny przykład nagrywania w Tkinterze.
- `requirements.txt` - zależności Pythona.
- `output/` - dane generowane przez aplikację.

## Uwagi Deweloperskie

Kilka miejsc w kodzie, od których warto zacząć zmiany:

- `MODEL_OPTIONS` w `app.py` - lista dostępnych modeli.
- `POSTPROCESS_REPLACEMENTS` w `app.py` - proste poprawki często mylonych fraz technicznych.
- `load_settings()` w `app.py` - domyślne ustawienia aplikacji.
- `ClipboardManager` - logika schowka dla X11/Wayland.
- `PasteManager` - logika auto-paste.

Transkrypcja i ładowanie modelu działają w tle, żeby nie blokować interfejsu. Wyniki są przekazywane do UI przez kolejkę `transcription_queue`.
