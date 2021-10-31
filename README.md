URDF Viewer using brax
=========

This is simple [URDF XML file](http://wiki.ros.org/en/urdf/Tutorials) viewer using [BRAX](https://github.com/google/brax).

Install
-------

```
$ pip install git+https://github.com/google/brax.git@main
```

NOTE: `pip install brax` may be old to run this script.

How to use
----------

```
$ urdf-viewer.py --xml_model_path=sample/test.urdf
```

Then, open  http://127.0.0.1:8000/ in your browser.