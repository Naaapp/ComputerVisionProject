# Computer Vision Project

Link to the [course] and to its [general description] on the Ulg website.

# 1. Part 1 
## 1.2 Edge points extraction

Following is the table containing the different methods used for edge points extraction.

| Author | Method | Description | Useful links |
| ------ | ------ | ------ | ------ |
| Bardhyl |  |  |  |
| Joachim | Morphological gradient (Gradient of Beucher) |  | [wikipedia : morphological gradient] <br> [OpenCV : morphological gradient]|
| Julien | Prewitt operator <br> (FDoG Coherent Line Drawing) |  | [Wikipedia : Prewit] <br> [Edge detection comparing methods] <br> [PDF : ? ]|
| Quentin |  |  |  |
| Théo | Canny Algorithm |  | [wikipedia : canny edge detector] <br> [OpenCV : canny edge detector] |

## 1.3 Segments and endpoints detection

Following is the table containing the different methods used for edge points extraction.

| Group | Method | Description | Useful links |
| ------ | ------ | ------ | ------ |
| Bardhyl <br> Julien <br> Quentin |  |  |  |
| Joachim <br> Théo | Probabilistic Hough Transform |  | |

# 2. Additional instructions about the project

- **Plagiarism** : Mr. Van Droogenbroeck was very clear that hee takes it very seriously. It is ok to take code (or to be inspired by code) from the internet but he expects us to  mention the authors. He will take STRONG actions against plagiarism (zero from the whole group for example...) .
- **Code** : There will be constraint on the environnment used for the project.
- **Report** : It should at most be 6 pages long. We need to keep it short.
- **Presentation** : We can do a demo at the presentation => I think he would give us more points if we did.
- **Part 1 report** : We don't need to do a performance evaluation in this part.

# 3. How to install pylsd binding

```bash
git clone https://github.com/primetang/pylsd.git
cd pylsd
nano ./pylsd/lsd.py
```

- With **nano**, we need to change line 15 and 20 provoking an error with tmp file name. Down below, there is the deprecated line (in comment) above the correct one

```python
# temp = os.path.abspath(str(np.random.randint(1, 1000000)) + 'ntl.txt').replace('\\', '/')
temp = os.path.abspath(str(np.random.randint(1, 1000000)) + 'ntl.txt').replace('\\', '/').encode('utf-8')

# lsdlib.lsdGet(src, ctypes.c_int(rows), ctypes.c_int(cols), temp)
lsdlib.lsdGet(src, ctypes.c_int(rows), ctypes.c_int(cols), ctypes.c_char_p(temp))
```

- Then we can launch the python script for installation

```bash
path/to/Annaconda/environement/python setup.py install
```

- If anaconda environment variables are not used, the default 'python' command is enough

- The openCV example from the git (**note**: this is in python 2.7, xrange has been replace by range in python 3)

```python
import cv2
import numpy as np
import os
from pylsd.lsd import lsd
fullName = 'car.jpg'
folder, imgName = os.path.split(fullName)
src = cv2.imread(fullName, cv2.IMREAD_COLOR)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
lines = lsd(gray)
for i in xrange(lines.shape[0]):
    pt1 = (int(lines[i, 0]), int(lines[i, 1]))
    pt2 = (int(lines[i, 2]), int(lines[i, 3]))
    width = lines[i, 4]
    cv2.line(src, pt1, pt2, (0, 0, 255), int(np.ceil(width / 2)))
cv2.imwrite(os.path.join(folder, 'cv2_' + imgName.split('.')[0] + '.jpg'), src)
```



[//]: # "Below is the list of references"

[course]: <https://orbi.uliege.be/handle/2268/184667>
[general description]: <https://www.programmes.uliege.be/cocoon/20182019/en/cours/ELEN0016-2.html>
[wikipedia : morphological gradient]:<https://en.wikipedia.org/wiki/Morphological_gradient>
[OpenCV : morphological gradient]:<https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html>
[wikipedia : canny edge detector]:<https://en.wikipedia.org/wiki/Canny_edge_detector>
[OpenCV : canny edge detector]:<https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html>
[Paper : Hough Transform Variant]:<https://jivp-eurasipjournals.springeropen.com/articles/10.1186/s13640-017-0180-7>
[Wikipedia : Prewit]:<https://en.wikipedia.org/wiki/Prewitt_operator>
[Edge detection comparing methods]:<https://medium.com/@nikatsanka/comparing-edge-detection-methods-638a2919476e>
[PDF : ? ]:<https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.108.559&rep=rep1&type=pdf>
