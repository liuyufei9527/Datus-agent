# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus/utils/class_utils.py."""

from datus.utils.class_utils import get_public_instance_methods


class TestGetPublicInstanceMethods:
    def test_basic_class(self):
        class MyClass:
            def public_method(self):
                pass

            def another_public(self):
                pass

            def _private_method(self):
                pass

            def __dunder(self):
                pass

        methods = get_public_instance_methods(MyClass)
        assert "public_method" in methods
        assert "another_public" in methods
        assert "_private_method" not in methods
        assert "__dunder" not in methods

    def test_excludes_staticmethod(self):
        class MyClass:
            @staticmethod
            def static_method():
                pass

            def instance_method(self):
                pass

        methods = get_public_instance_methods(MyClass)
        assert "instance_method" in methods
        assert "static_method" not in methods

    def test_excludes_classmethod(self):
        class MyClass:
            @classmethod
            def class_method(cls):
                pass

            def instance_method(self):
                pass

        methods = get_public_instance_methods(MyClass)
        assert "instance_method" in methods
        assert "class_method" not in methods

    def test_empty_class(self):
        class EmptyClass:
            pass

        methods = get_public_instance_methods(EmptyClass)
        assert len(methods) == 0

    def test_returns_dict(self):
        class MyClass:
            def method_a(self):
                pass

        methods = get_public_instance_methods(MyClass)
        assert isinstance(methods, dict)
        assert callable(methods["method_a"])
