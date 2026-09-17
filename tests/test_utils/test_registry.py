import mmcv2
import pytest
from mmcv2.cnn import MODELS, build_model_from_cfg
from torch import nn


def test_registry():
    CATS = mmcv2.Registry("cat")
    assert CATS.name == "cat"
    assert CATS.module_dict == {}
    assert len(CATS) == 0

    @CATS.register_module()
    class BritishShorthair:
        pass

    assert len(CATS) == 1
    assert CATS.get("BritishShorthair") is BritishShorthair

    class Munchkin:
        pass

    CATS.register_module(module=Munchkin)
    assert len(CATS) == 2
    assert CATS.get("Munchkin") is Munchkin
    assert "Munchkin" in CATS

    with pytest.raises(KeyError):
        CATS.register_module(module=Munchkin)

    CATS.register_module(module=Munchkin, force=True)
    assert len(CATS) == 2

    # force=False
    with pytest.raises(KeyError):

        @CATS.register_module()
        class BritishShorthair:
            pass

    @CATS.register_module(force=True)
    class BritishShorthair:
        pass

    assert len(CATS) == 2

    assert CATS.get("PersianCat") is None
    assert "PersianCat" not in CATS

    @CATS.register_module(name=["Siamese", "Siamese2"])
    class SiameseCat:
        pass

    assert CATS.get("Siamese").__name__ == "SiameseCat"
    assert CATS.get("Siamese2").__name__ == "SiameseCat"

    class SphynxCat:
        pass

    CATS.register_module(name="Sphynx", module=SphynxCat)
    assert CATS.get("Sphynx") is SphynxCat

    CATS.register_module(name=["Sphynx1", "Sphynx2"], module=SphynxCat)
    assert CATS.get("Sphynx2") is SphynxCat

    repr_str = "Registry(name=cat, items={"
    repr_str += "'BritishShorthair': <class 'test_registry.test_registry.<locals>.BritishShorthair'>, "
    repr_str += "'Munchkin': <class 'test_registry.test_registry.<locals>.Munchkin'>, "
    repr_str += "'Siamese': <class 'test_registry.test_registry.<locals>.SiameseCat'>, "
    repr_str += "'Siamese2': <class 'test_registry.test_registry.<locals>.SiameseCat'>, "
    repr_str += "'Sphynx': <class 'test_registry.test_registry.<locals>.SphynxCat'>, "
    repr_str += "'Sphynx1': <class 'test_registry.test_registry.<locals>.SphynxCat'>, "
    repr_str += "'Sphynx2': <class 'test_registry.test_registry.<locals>.SphynxCat'>"
    repr_str += "})"
    assert repr(CATS) == repr_str

    # name type
    with pytest.raises(TypeError):
        CATS.register_module(name=7474741, module=SphynxCat)

    # the registered module should be a class
    with pytest.raises(TypeError):
        CATS.register_module(0)

    @CATS.register_module()
    def muchkin():
        pass

    assert CATS.get("muchkin") is muchkin
    assert "muchkin" in CATS

    # can only decorate a class or a function
    with pytest.raises(TypeError):

        class Demo:
            def some_method(self):
                pass

        method = Demo().some_method
        CATS.register_module(name="some_method", module=method)


def test_multi_scope_registry():
    DOGS = mmcv2.Registry("dogs")
    assert DOGS.name == "dogs"
    assert DOGS.scope == "test_registry"
    assert DOGS.module_dict == {}
    assert len(DOGS) == 0

    @DOGS.register_module()
    class GoldenRetriever:
        pass

    assert len(DOGS) == 1
    assert DOGS.get("GoldenRetriever") is GoldenRetriever

    HOUNDS = mmcv2.Registry("dogs", parent=DOGS, scope="hound")

    @HOUNDS.register_module()
    class BloodHound:
        pass

    assert len(HOUNDS) == 1
    assert HOUNDS.get("BloodHound") is BloodHound
    assert DOGS.get("hound.BloodHound") is BloodHound
    assert HOUNDS.get("hound.BloodHound") is BloodHound

    LITTLE_HOUNDS = mmcv2.Registry("dogs", parent=HOUNDS, scope="little_hound")

    @LITTLE_HOUNDS.register_module()
    class Dachshund:
        pass

    assert len(LITTLE_HOUNDS) == 1
    assert LITTLE_HOUNDS.get("Dachshund") is Dachshund
    assert LITTLE_HOUNDS.get("hound.BloodHound") is BloodHound
    assert HOUNDS.get("little_hound.Dachshund") is Dachshund
    assert DOGS.get("hound.little_hound.Dachshund") is Dachshund

    MID_HOUNDS = mmcv2.Registry("dogs", parent=HOUNDS, scope="mid_hound")

    @MID_HOUNDS.register_module()
    class Beagle:
        pass

    assert MID_HOUNDS.get("Beagle") is Beagle
    assert HOUNDS.get("mid_hound.Beagle") is Beagle
    assert DOGS.get("hound.mid_hound.Beagle") is Beagle
    assert LITTLE_HOUNDS.get("hound.mid_hound.Beagle") is Beagle
    assert MID_HOUNDS.get("hound.BloodHound") is BloodHound
    assert MID_HOUNDS.get("hound.Dachshund") is None


def test_build_from_cfg():
    BACKBONES = mmcv2.Registry("backbone")

    @BACKBONES.register_module()
    class ResNet:
        def __init__(self, depth, stages=4):
            self.depth = depth
            self.stages = stages

    @BACKBONES.register_module()
    class ResNeXt:
        def __init__(self, depth, stages=4):
            self.depth = depth
            self.stages = stages

    cfg = {"type": "ResNet", "depth": 50}
    model = mmcv2.build_from_cfg(cfg, BACKBONES)
    assert isinstance(model, ResNet)
    assert model.depth == 50 and model.stages == 4

    cfg = {"type": "ResNet", "depth": 50}
    model = mmcv2.build_from_cfg(cfg, BACKBONES, default_args={"stages": 3})
    assert isinstance(model, ResNet)
    assert model.depth == 50 and model.stages == 3

    cfg = {"type": "ResNeXt", "depth": 50, "stages": 3}
    model = mmcv2.build_from_cfg(cfg, BACKBONES)
    assert isinstance(model, ResNeXt)
    assert model.depth == 50 and model.stages == 3

    cfg = {"type": ResNet, "depth": 50}
    model = mmcv2.build_from_cfg(cfg, BACKBONES)
    assert isinstance(model, ResNet)
    assert model.depth == 50 and model.stages == 4

    # type defined using default_args
    cfg = {"depth": 50}
    model = mmcv2.build_from_cfg(cfg, BACKBONES, default_args={"type": "ResNet"})
    assert isinstance(model, ResNet)
    assert model.depth == 50 and model.stages == 4

    cfg = {"depth": 50}
    model = mmcv2.build_from_cfg(cfg, BACKBONES, default_args={"type": ResNet})
    assert isinstance(model, ResNet)
    assert model.depth == 50 and model.stages == 4

    # non-registered class
    with pytest.raises(KeyError):
        cfg = {"type": "VGG"}
        model = mmcv2.build_from_cfg(cfg, BACKBONES)

    # cfg['type'] should be a str or class
    with pytest.raises(TypeError):
        cfg = {"type": 1000}
        model = mmcv2.build_from_cfg(cfg, BACKBONES)

    # cfg should contain the key "type"
    with pytest.raises(KeyError, match='must contain the key "type"'):
        cfg = {"depth": 50, "stages": 4}
        model = mmcv2.build_from_cfg(cfg, BACKBONES)

    # cfg or default_args should contain the key "type"
    with pytest.raises(KeyError, match='must contain the key "type"'):
        cfg = {"depth": 50}
        model = mmcv2.build_from_cfg(cfg, BACKBONES, default_args={"stages": 4})

    # incorrect arguments
    with pytest.raises(TypeError):
        cfg = {"type": "ResNet", "non_existing_arg": 50}
        model = mmcv2.build_from_cfg(cfg, BACKBONES)


def test_model_registry_uses_its_specialized_builder():
    backbones = mmcv2.Registry("backbone", build_func=build_model_from_cfg)

    @backbones.register_module()
    class ResNet(nn.Module):
        def __init__(self, depth, stages=4):
            super().__init__()
            self.depth = depth
            self.stages = stages

        def forward(self, x):
            return x

    @backbones.register_module()
    class ResNeXt(ResNet):
        pass

    models = backbones.build([
        {"type": "ResNet", "depth": 50},
        {"type": "ResNeXt", "depth": 50, "stages": 3},
    ])
    assert isinstance(models, nn.Sequential)
    assert (models[0].depth, models[0].stages) == (50, 4)
    assert (models[1].depth, models[1].stages) == (50, 3)

    inherited_models = mmcv2.Registry("models", parent=MODELS, scope="new")
    assert inherited_models.build_func is build_model_from_cfg

    def custom_builder(cfg):
        return cfg

    assert mmcv2.Registry("models", parent=MODELS, build_func=custom_builder).build_func is custom_builder
