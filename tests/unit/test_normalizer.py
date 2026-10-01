from unittest.mock import MagicMock

from app.normalizer import normalize_entry, _strip_cdata, _strip_html, _is_noema_wrapper


def _make_entry(title="Test Title", link="https://example.com/article", summary="Test summary", author="Author"):
    entry = MagicMock()
    entry.title = title
    entry.link = link
    entry.summary = summary
    entry.author = author
    entry.published = None
    entry.updated = None
    entry.created = None
    entry.published_parsed = None
    entry.updated_parsed = None
    entry.created_parsed = None
    return entry


class TestStripCdata:
    def test_plain_title_unchanged(self):
        assert _strip_cdata("Hello World") == "Hello World"

    def test_cdata_wrapped_title(self):
        assert _strip_cdata("<![CDATA[Hello World]]>") == "Hello World"

    def test_cdata_with_accents(self):
        assert _strip_cdata("<![CDATA[Así terminó el medallero]]>") == "Así terminó el medallero"

    def test_cdata_with_inner_markup_ignored(self):
        result = _strip_cdata("<![CDATA[Text with <b>bold</b>]]>")
        assert result == "Text with <b>bold</b>"

    def test_cdata_with_newlines(self):
        result = _strip_cdata("<![CDATA[Line one\nLine two]]>")
        assert result == "Line one\nLine two"

    def test_partial_cdata_not_stripped(self):
        assert _strip_cdata("Before <![CDATA[inner]]> After") == "Before <![CDATA[inner]]> After"

    def test_empty_string(self):
        assert _strip_cdata("") == ""


class TestStripHtml:
    def test_plain_text_unchanged(self):
        assert _strip_html("Hello World") == "Hello World"

    def test_paragraph_tags(self):
        assert _strip_html("<p>Hello World</p>") == "Hello World"

    def test_link_preserves_text(self):
        result = _strip_html('<a href="https://example.com">click here</a>')
        assert result == "click here"

    def test_link_with_surrounding_text(self):
        result = _strip_html("See <a href='https://example.com'>this article</a> for details.")
        assert "this article" in result
        assert "See" in result
        assert "details" in result

    def test_multiple_paragraphs(self):
        result = _strip_html("<p>First</p><p>Second</p>")
        assert "First" in result
        assert "Second" in result

    def test_html_entities(self):
        assert _strip_html("A &amp; B") == "A & B"

    def test_html_entities_quotes(self):
        assert _strip_html("&quot;quoted&quot;") == '"quoted"'

    def test_html_entity_nbsp(self):
        result = _strip_html("word1&nbsp;word2")
        assert "word1" in result and "word2" in result

    def test_nested_tags(self):
        result = _strip_html("<div><p>Hello <b>world</b></p></div>")
        assert "Hello" in result
        assert "world" in result

    def test_excessive_whitespace_normalized(self):
        result = _strip_html("<p>Hello    world</p>")
        assert "Hello world" in result

    def test_newlines_in_html(self):
        result = _strip_html("<p>Hello\n\nworld</p>")
        assert "Hello" in result
        assert "world" in result

    def test_img_tag_stripped(self):
        result = _strip_html('Text <img src="pic.jpg" alt="photo"> more')
        assert "Text" in result
        assert "more" in result
        assert "<img" not in result


class TestIsNoemaWrapper:
    def test_noema_wrapper_detected(self):
        text = (
            'The post Title appeared first on NOEMA.'
        )
        assert _is_noema_wrapper(text) is True

    def test_noema_wrapper_without_period(self):
        text = (
            'The post Title appeared first on NOEMA'
        )
        assert _is_noema_wrapper(text) is True

    def test_noema_wrapper_with_space_before_period(self):
        text = (
            'The post Title appeared first on NOEMA .'
        )
        assert _is_noema_wrapper(text) is True

    def test_legitimate_summary_not_detected(self):
        assert _is_noema_wrapper("This is a real article summary about technology.") is False

    def test_empty_string_not_detected(self):
        assert _is_noema_wrapper("") is False

    def test_partial_match_not_detected(self):
        assert _is_noema_wrapper("The post appeared first on something else.") is False


class TestNormalizeEntry:
    def test_normal_entry(self):
        entry = _make_entry()
        article = normalize_entry(entry, "TestSource")
        assert article is not None
        assert article.title == "Test Title"
        assert article.url == "https://example.com/article"
        assert article.source == "TestSource"
        assert article.summary == "Test summary"
        assert article.author == "Author"

    def test_missing_title_returns_none(self):
        entry = _make_entry()
        entry.title = None
        assert normalize_entry(entry, "TestSource") is None

    def test_missing_link_returns_none(self):
        entry = _make_entry()
        entry.link = None
        assert normalize_entry(entry, "TestSource") is None

    def test_empty_title_returns_none(self):
        entry = _make_entry(title="")
        assert normalize_entry(entry, "TestSource") is None

    def test_missing_summary_defaults_to_empty(self):
        entry = _make_entry(summary=None)
        article = normalize_entry(entry, "TestSource")
        assert article is not None
        assert article.summary == ""

    def test_missing_author_defaults_to_none(self):
        entry = _make_entry(author=None)
        article = normalize_entry(entry, "TestSource")
        assert article.author is None

    def test_whitespace_is_stripped(self):
        entry = _make_entry(title="  Title  ", summary="  Summary  ")
        article = normalize_entry(entry, "TestSource")
        assert article.title == "Title"
        assert article.summary == "Summary"

    def test_fetched_at_is_set(self):
        article = normalize_entry(_make_entry(), "TestSource")
        assert article.fetched_at is not None

    def test_cdata_title_is_cleaned(self):
        entry = _make_entry(title="<![CDATA[Hello World]]>")
        article = normalize_entry(entry, "TestSource")
        assert article.title == "Hello World"

    def test_cdata_title_with_accents(self):
        entry = _make_entry(title="<![CDATA[México: terremoto en Colômbia]]>")
        article = normalize_entry(entry, "TestSource")
        assert article.title == "México: terremoto en Colômbia"

    def test_html_summary_cleaned(self):
        entry = _make_entry(summary="<p>Hello <b>world</b></p>")
        article = normalize_entry(entry, "TestSource")
        assert "<p>" not in article.summary
        assert "<b>" not in article.summary
        assert "Hello" in article.summary
        assert "world" in article.summary

    def test_html_summary_with_link_preserves_text(self):
        entry = _make_entry(summary='<p>See <a href="https://example.com">this article</a>.</p>')
        article = normalize_entry(entry, "TestSource")
        assert "this article" in article.summary
        assert "<a" not in article.summary

    def test_html_entities_decoded(self):
        entry = _make_entry(summary="A &amp; B &lt; C")
        article = normalize_entry(entry, "TestSource")
        assert "A & B" in article.summary
        assert "< C" in article.summary

    def test_multiple_paragraphs_normalized(self):
        entry = _make_entry(summary="<p>First paragraph.</p><p>Second paragraph.</p>")
        article = normalize_entry(entry, "TestSource")
        assert "First paragraph" in article.summary
        assert "Second paragraph" in article.summary

    def test_excessive_whitespace_normalized(self):
        entry = _make_entry(summary="Hello    world   how   are   you")
        article = normalize_entry(entry, "TestSource")
        assert "Hello world how are you" in article.summary

    def test_noema_wrapper_summary_becomes_empty(self):
        entry = _make_entry(
            summary=(
                '<p>The post <a href="https://example.com">Title</a> '
                'appeared first on <a href="https://www.noemamag.com">NOEMA</a>.</p>'
            )
        )
        article = normalize_entry(entry, "TestSource")
        assert article.summary == ""

    def test_noema_wrapper_with_entities(self):
        entry = _make_entry(
            summary=(
                '<p>The post <a href="https://example.com">AI &#8216;Control&#8217; Transition</a> '
                'appeared first on <a href="https://www.noemamag.com">NOEMA</a>.</p>'
            )
        )
        article = normalize_entry(entry, "TestSource")
        assert article.summary == ""

    def test_empty_summary_stays_empty(self):
        entry = _make_entry(summary="")
        article = normalize_entry(entry, "TestSource")
        assert article.summary == ""

    def test_plain_text_summary_unchanged(self):
        entry = _make_entry(summary="This is a plain text summary.")
        article = normalize_entry(entry, "TestSource")
        assert article.summary == "This is a plain text summary."

    def test_search_still_works_after_html_strip(self):
        entry = _make_entry(
            title="Test Article",
            summary="<p>This article discusses <b>technology</b> in <a href='https://example.com'>Mexico</a>.</p>"
        )
        article = normalize_entry(entry, "TestSource")
        assert "technology" in article.summary.lower()
        assert "mexico" in article.summary.lower()
