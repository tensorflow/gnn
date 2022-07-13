# Scope &amp; Focus

For this initial release, and likely for the next several releases, we will be
driven by internal, production needs and will therefore need to prioritize our
internal users requirements. However, we want to encourage that you submit
contributions, either as issues or PRs within this proejct or as an RFC in
the [TensorFlow Community RFC](https://github.com/tensorflow/community/tree/master/rfcs)
repo.

# How to Contribute

We'd love to have your contributions to this project! There are
just a few small guidelines you need to follow.

### Step 1. Open an issue

Before making any changes, we recommend opening an issue (if one doesn't already
exist) and discussing your proposed changes. This way, we can give you feedback
and validate the proposed changes.

If the changes are minor (simple bug fix or documentation fix), then feel free
to open a PR without discussion.

### Step 2. Make code changes

To make code changes, you need to fork the repository. You will need to setup a
development environment and run the unit tests. This is covered in the
[developer guide](https://github.com/tensorflow/gnn/tensorflow_gnn/docs/guide/developer.md).

### Step 3. Create a pull request

Once the change is ready, open a pull request from your branch in your fork to
the main branch in [tensorflow/gnn](https://github.com/tensorflow/gnn).

### Step 4. Sign the Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution;
this simply gives us permission to use and redistribute your contributions as
part of the project.

After creating a pull request, the `google-cla` bot will comment on your pull
request with instructions on signing the Contributor License Agreement (CLA) if
you haven't done so. Please follow the instructions to sign the CLA. A `cla:yes`
tag is then added to the pull request.

Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one. You generally only need to
submit a CLA once, so if you've already submitted one (even if it was for a
different project), you probably don't need to do it again.

### Step 5. Code review

A reviewer will review the pull request and provide comments. The reviewer may
add a `kokoro:force-run` label to trigger the continuous integration tests.

If the tests fail, look into the error messages and try to fix them.

There may be
several rounds of comments and code changes before the pull request gets
approved by the reviewer.

### Step 6. Merging

Once the pull request is approved, a `ready to pull` tag will be added to the
pull request. A team member will take care of the merging.

Here is an [example pull request](https://github.com/tensorflow/gnn/pull/37)
for your reference.

## Community Guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google.com/conduct/).
