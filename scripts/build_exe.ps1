param(
    [string]$Name = "anasens-cli"
)

python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pyinstaller
pyinstaller --onefile --name $Name src/anasens/interfaces/cli.py
