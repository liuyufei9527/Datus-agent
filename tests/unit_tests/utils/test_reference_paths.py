# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus/utils/reference_paths.py."""

import pytest

from datus.utils.reference_paths import (
    normalize_reference_path,
    quote_path_segment,
    split_reference_path,
)


class TestNormalizeReferencePath:
    def test_empty_string(self):
        assert normalize_reference_path("") == ""

    def test_none_like(self):
        assert normalize_reference_path(None) == ""

    def test_simple_path(self):
        assert normalize_reference_path("domain.layer.name") == "domain.layer.name"

    def test_strip_trailing_punctuation(self):
        assert normalize_reference_path("domain.layer.name.") == "domain.layer.name"
        assert normalize_reference_path("domain.layer.name,") == "domain.layer.name"
        assert normalize_reference_path("domain.layer.name;") == "domain.layer.name"

    def test_quoted_last_segment(self):
        result = normalize_reference_path('domain.layer."name with spaces"')
        assert result == "domain.layer.name with spaces"

    def test_whitespace_trimming(self):
        assert normalize_reference_path("  domain.name  ") == "domain.name"

    def test_quoted_with_whitespace_outside(self):
        result = normalize_reference_path('"quoted name" extra text')
        assert result == "quoted name"

    def test_all_quoted(self):
        result = normalize_reference_path('"just quoted"')
        assert result == "just quoted"

    def test_empty_after_cleaning(self):
        # Only punctuation
        assert normalize_reference_path("...") == ""

    def test_multiple_segments_quoted(self):
        result = normalize_reference_path('"first"."second"')
        # Only last segment is unquoted
        assert result == '"first".second'


class TestSplitReferencePath:
    def test_empty_string(self):
        assert split_reference_path("") == []

    def test_none(self):
        assert split_reference_path(None) == []

    def test_whitespace_only(self):
        assert split_reference_path("   ") == []

    def test_simple_path(self):
        assert split_reference_path("domain.layer.name") == ["domain", "layer", "name"]

    def test_quoted_segments(self):
        result = split_reference_path('domain."layer one"."name two"')
        assert result == ["domain", "layer one", "name two"]

    def test_empty_segments_between_dots(self):
        result = split_reference_path("domain..name")
        assert result == ["domain", "name"]

    def test_trailing_punctuation(self):
        result = split_reference_path("domain.name.")
        assert result == ["domain", "name"]

    def test_whitespace_outside_quotes(self):
        result = split_reference_path('"first part" other text')
        assert result == ["first part"]

    def test_single_unquoted(self):
        result = split_reference_path("simple")
        assert result == ["simple"]


class TestQuotePathSegment:
    def test_empty_segment(self):
        assert quote_path_segment("") == ""

    def test_simple_name(self):
        assert quote_path_segment("simple") == "simple"

    def test_name_with_space(self):
        assert quote_path_segment("name with space") == '"name with space"'

    def test_name_with_dot(self):
        assert quote_path_segment("name.dot") == '"name.dot"'

    def test_name_with_at(self):
        assert quote_path_segment("name@symbol") == '"name@symbol"'

    def test_already_quoted(self):
        result = quote_path_segment('"already quoted"')
        # Quotes are stripped first, then content with space gets re-quoted
        assert result == '"already quoted"'

    def test_internal_quotes_removed(self):
        result = quote_path_segment('na"me')
        assert result == "name"

    def test_whitespace_only(self):
        assert quote_path_segment("   ") == ""

    def test_tab_in_name(self):
        assert quote_path_segment("name\there").startswith('"')
