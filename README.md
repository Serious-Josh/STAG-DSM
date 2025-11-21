A DSM-FER system to detect fatigue in a driver. Utilizes a custom architecture named STAG.

dsm.py is the runnable version and requires correct dataset root setup. [Google Drive](https://drive.google.com/drive/folders/1IUQ5uUj5Mw7sXoiluSbwrMNoMlmJ1Iih?usp=sharing) link.

Directory structure must be as given here:
```
/root
  dsm_file.py
  /datasets
    /affectNet
    /raf-db
    /DMD
```

dsm_superVerified.py is a very heavy debug version of the script and is verified to be runnable. It is the version all development and testing was performed with.
Any other version is not to be ran and is used for demonstration purposes only. They're ability to be ran has not been verified.

Install requirements.txt to virtual environment using
```pip install -r requirements.txt```

When running, first run
```python dsm_file.py --dmd-export```
to generate proper csv and mapping files for the dmd dataset.

Then you can run
```python dsm_file.py --train```
to properly train the model.
