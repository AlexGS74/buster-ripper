"""Request shapes and stamp sources for the image tests, so no module builds them twice.

`--strip-images` reads `messages` and rewrites the image blocks it finds, so every
witness needs a body with user turns, assistant turns, and image blocks carrying a
`source` mapping. The writer mutates the body it is handed, so a case that wants two
requests needs two bodies — that is what `one_image()` is for, and why these builders
return fresh structures rather than sharing one.

Import it (`from tests.image_bodies import one_image`) the way the route tests import
`tests.upstream_stub`; do not wrap these in fixtures, which only costs every test an
extra parameter. The two things that do need the `mocker` a test already has are
`stamp_source()` and `reset_cache()`, since they patch module state.
"""

from buster_ripper import image


def image_block(source):
    return {"type": "image", "source": source}


def text_block(text):
    return {"type": "text", "text": text}


def user(content):
    return {"role": "user", "content": content}


def assistant(content):
    return {"role": "assistant", "content": content}


def body(*messages):
    return {"messages": list(messages)}


def one_image(source):
    """A fresh single-image body — a new one per call, because the writer mutates."""
    return body(user([image_block(source)]))


def say(text):
    """A turn with list content that holds no image.

    The position table's text-only turns are list content on purpose: a plain string
    sits outside the writer's visit scope, so a detector abbreviated to "the first
    message it looks at" would pass those rows by accident rather than by scope.
    """
    return [text_block(text)]


def _blocks(processed):
    """Every content block of every message with list content."""
    for msg in processed.get("messages", []):
        if isinstance(msg, dict) and isinstance(msg.get("content"), list):
            yield from msg["content"]


def placeholders(processed):
    """Every placeholder key in a processed body, in the order the writer emitted them."""
    keys = []
    for block in _blocks(processed):
        text = block.get("text", "") if isinstance(block, dict) else ""
        if text.startswith("[image:") and text.endswith("]"):
            keys.append(text[len("[image:"):-1])
    return keys


def images_left(processed):
    """Image blocks still sitting in a body — for a stripped request, none."""
    return [b for b in _blocks(processed) if isinstance(b, dict) and b.get("type") == "image"]


def sources(session_id):
    """The payloads one session's cache holds, so a case can name an image by its source."""
    return [entry[0] for entry in image._image_cache.get(session_id, {}).values()]


def ids(session_id):
    """The `id` of each source one session holds — a readable set for eviction rows."""
    return {source.get("id") for source in sources(session_id)}


def reset_cache():
    """Drop every cached image, so a case states its own starting state (`DC-claim-289-r1`)."""
    image._image_cache.clear()


class Clock:
    """A monotonic stamp source the test moves, standing in for `image._now`."""

    def __init__(self, start=1000.0):
        self.value = start

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def stamp_source(mocker, start=1000.0):
    """Install a movable stamp source, so no TTL witness sleeps on a clock."""
    source = Clock(start)
    mocker.patch.object(image, "_now", source)
    return source
