import sys

import mmcv2
import pytest
from mmcv2.utils.misc import has_method


def test_to_ntuple():
    single_number = 2
    assert mmcv2.utils.to_1tuple(single_number) == (single_number,)
    assert mmcv2.utils.to_2tuple(single_number) == (single_number, single_number)
    assert mmcv2.utils.to_3tuple(single_number) == (
        single_number,
        single_number,
        single_number,
    )
    assert mmcv2.utils.to_4tuple(single_number) == (
        single_number,
        single_number,
        single_number,
        single_number,
    )
    assert mmcv2.utils.to_ntuple(5)(single_number) == (
        single_number,
        single_number,
        single_number,
        single_number,
        single_number,
    )
    assert mmcv2.utils.to_ntuple(6)(single_number) == (
        single_number,
        single_number,
        single_number,
        single_number,
        single_number,
        single_number,
    )


def test_iter_cast():
    assert mmcv2.list_cast([1, 2, 3], int) == [1, 2, 3]
    assert mmcv2.list_cast(["1.1", 2, "3"], float) == [1.1, 2.0, 3.0]
    assert mmcv2.list_cast([1, 2, 3], str) == ["1", "2", "3"]
    assert mmcv2.tuple_cast((1, 2, 3), str) == ("1", "2", "3")
    assert next(mmcv2.iter_cast([1, 2, 3], str)) == "1"
    with pytest.raises(TypeError):
        mmcv2.iter_cast([1, 2, 3], "")
    with pytest.raises(TypeError):
        mmcv2.iter_cast(1, str)


def test_is_seq_of():
    assert mmcv2.is_seq_of([1.0, 2.0, 3.0], float)
    assert mmcv2.is_seq_of([(1,), (2,), (3,)], tuple)
    assert mmcv2.is_seq_of((1.0, 2.0, 3.0), float)
    assert mmcv2.is_list_of([1.0, 2.0, 3.0], float)
    assert not mmcv2.is_seq_of((1.0, 2.0, 3.0), float, seq_type=list)
    assert not mmcv2.is_tuple_of([1.0, 2.0, 3.0], float)
    assert not mmcv2.is_seq_of([1.0, 2, 3], int)
    assert not mmcv2.is_seq_of((1.0, 2, 3), int)


def test_requires_package(capsys):

    @mmcv2.requires_package("nnn")
    def func_a():
        pass

    @mmcv2.requires_package(["numpy", "n1", "n2"])
    def func_b():
        pass

    @mmcv2.requires_package("numpy")
    def func_c():
        return 1

    with pytest.raises(RuntimeError):
        func_a()
    out, _ = capsys.readouterr()
    assert out == ('Prerequisites "nnn" are required in method "func_a" but not found, please install them first.\n')

    with pytest.raises(RuntimeError):
        func_b()
    out, _ = capsys.readouterr()
    assert out == ('Prerequisites "n1, n2" are required in method "func_b" but not found, please install them first.\n')

    assert func_c() == 1


def test_requires_executable(capsys):

    @mmcv2.requires_executable("nnn")
    def func_a():
        pass

    @mmcv2.requires_executable([sys.executable, "n1", "n2"])
    def func_b():
        pass

    @mmcv2.requires_executable(sys.executable)
    def func_c():
        return 1

    with pytest.raises(RuntimeError):
        func_a()
    out, _ = capsys.readouterr()
    assert out == ('Prerequisites "nnn" are required in method "func_a" but not found, please install them first.\n')

    with pytest.raises(RuntimeError):
        func_b()
    out, _ = capsys.readouterr()
    assert out == ('Prerequisites "n1, n2" are required in method "func_b" but not found, please install them first.\n')

    assert func_c() == 1


def test_import_modules_from_strings():
    # multiple imports
    import os.path as osp_
    import sys as sys_

    osp, sys = mmcv2.import_modules_from_strings(["os.path", "sys"])
    assert osp == osp_
    assert sys == sys_

    # single imports
    osp = mmcv2.import_modules_from_strings("os.path")
    assert osp == osp_
    # No imports
    assert mmcv2.import_modules_from_strings(None) is None
    assert mmcv2.import_modules_from_strings([]) is None
    assert mmcv2.import_modules_from_strings("") is None
    # Unsupported types
    with pytest.raises(TypeError):
        mmcv2.import_modules_from_strings(1)
    with pytest.raises(TypeError):
        mmcv2.import_modules_from_strings([1])
    # Failed imports
    with pytest.raises(ImportError):
        mmcv2.import_modules_from_strings("_not_implemented_module")
    imported = mmcv2.import_modules_from_strings("_not_implemented_module", allow_failed_imports=True)
    assert imported is None
    imported = mmcv2.import_modules_from_strings(["os.path", "_not_implemented"], allow_failed_imports=True)
    assert imported[0] == osp
    assert imported[1] is None


def test_is_method_overridden():

    class Base:
        def foo1():
            pass

        def foo2():
            pass

    class Sub(Base):
        def foo1():
            pass

    # test passing sub class directly
    assert mmcv2.is_method_overridden("foo1", Base, Sub)
    assert not mmcv2.is_method_overridden("foo2", Base, Sub)

    # test passing instance of sub class
    sub_instance = Sub()
    assert mmcv2.is_method_overridden("foo1", Base, sub_instance)
    assert not mmcv2.is_method_overridden("foo2", Base, sub_instance)

    # base_class should be a class, not instance
    base_instance = Base()
    with pytest.raises(AssertionError):
        mmcv2.is_method_overridden("foo1", base_instance, sub_instance)


def test_has_method():

    class Foo:
        def __init__(self, name):
            self.name = name

        def print_name(self):
            print(self.name)

    foo = Foo("foo")
    assert not has_method(foo, "name")
    assert has_method(foo, "print_name")
