VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`tests/test_image.py` asserts `has_images` is False for the earliest-image, latest-text-only body

## What it actually does
`TestImageDetection::test_has_images_checks_last_user_message` (L49-58) builds exactly that
body and asserts False:

- L53 `{"role": "user", "content": [{"type": "image", "source": {}}]}` — the image is in the
  *earliest* message of the conversation, and is the only image in it.
- L54 `{"role": "assistant", "content": "response"}`
- L55 `{"role": "user", "content": "no image here"}` — the *latest* user message is text-only.
- L58 `assert image.has_images(body) is False`

It is the only False-asserting `has_images` case in the file whose body contains an image at
all: L23-24 (no messages), L29 (no user message) and L34 (single text-only user message) all
have no image anywhere. So the False is specifically the earliest-image / latest-text-only
case, matching the claim.

## Source
/Users/alexhubitski/Projects/buster-ripper/tests/test_image.py L49-58 (whole file read; the
claim's subject is the test suite, so the test file is the production surface for it — P10
exception)

## Source Version
tests/test_image.py, 286 lines, content identified by the fingerprint below; repository
revision not consulted (Git metadata is out of scope for this role)

## Source Fingerprint
sha256:b603bfd1624e30eab1c6d57390194a71cfd3716525abc33c742292bc5a3dfda0

## Invalidates On
- Reordering or editing L51-57 so the image moves into the latest user message, or so the
  earliest user message becomes text-only.
- Changing L58's expected value to True, or renaming/deleting
  `test_has_images_checks_last_user_message`.
- Adding a second assertion of the same shape with a different expected result, which would
  make "the earliest-image, latest-text-only body" ambiguous.

## Checked At
2026-10-05T18:52:58Z
