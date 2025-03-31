from setuptools import setup
from Cython.Build import cythonize
import numpy as np

file = input('Pyx file path: ')

setup(
    ext_modules=cythonize(file),
    include_dirs=[np.get_include()]
)

# python setup.py build_ext --inplace

# メモ: Cythonで高速化するために(2025/03/31)
# Cythonによる高速化の恩恵はループ処理の高速化によるものである
# numpy関数やPython由来の関数の呼び出しはできるだけ少なくし、C言語やC++の関数で代用あるいは自作できるならそうするべき
# 自作関数を用いたとしても、引数に取るオブジェクトが純粋Python由来である場合、処理が遅くなる点には注意が必要
# 可能な限りオブジェクトの生成も純粋Pythonを使わずに行う
# Cythonコード(.pyx)内だけで呼び出す関数はcpdefではなくcdefを使う
# 関数はできるだけ単純な四則演算に落とし込む
# numpyのndarrayではなくmemoryviewを使い、配列にはインデックスアクセスを徹底する
# できるだけappendではなく、最初に配列を生成し、その各要素にアクセスする形で最終的な配列を作る