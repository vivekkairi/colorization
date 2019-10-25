# Colorization

A flask web service to colorize gray scale images using Convolutional Neural Networks

## Installation

```bash
pip install -r requirements.txt
```

## Samples
![Grayscale lady image](https://github.com/vivekkairi/colorization/blob/master/static/processed_png/lady.png)
![RGB lady image](https://github.com/vivekkairi/colorization/blob/master/static/colored_img/lady.png)

![Grayscale bird](https://github.com/vivekkairi/colorization/blob/master/static/processed_png/bird.png)
![Colorized bird](https://github.com/vivekkairi/colorization/blob/master/static/colored_img/bird.png)

![Grayscale girl](https://github.com/vivekkairi/colorization/blob/master/static/processed_png/girl.png)
![Grayscale girl](https://github.com/vivekkairi/colorization/blob/master/static/colored_img/girl.png)

## Usage

Open app.py and change UPLOAD_FOLDER
```python
UPLOAD_FOLDER = '/path/to/files'
```
Run the program using 
```bash
python3 app.py
```

## References
[deep-koalarization](https://github.com/baldassarreFe/deep-koalarization/)

[Freecodecamp](https://www.freecodecamp.org/news/colorize-b-w-photos-with-a-100-line-neural-network-53d9b4449f8d/)

## License
[MIT](https://choosealicense.com/licenses/mit/)
