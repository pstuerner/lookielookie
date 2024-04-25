from datetime import datetime as dt
from lookielookie import Backbone
import typer

app = typer.Typer()


@app.command()
def timeseries():
    weekday = dt.now().weekday()
    if weekday in [6, 0]: return
    
    bb = Backbone()
    bb.ohlcv()
    bb.indicators()
    bb.signals()

@app.command()
def fundamentals():
    bb = Backbone()
    bb.fundamentals()

def main():
    app()

if __name__ == "__main__":
    app()
