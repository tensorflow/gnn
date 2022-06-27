"""Version info.

Releases of the TF-GNN library use [Semantic Versioning](https://semver.org/)
with the version string syntax of [PEP 440](https://peps.python.org/pep-0440/):

  * A release has a version MAJOR.MINOR.PATCH with three numeric parts
    and the conventional meaning regarding API stability. Recall that
    MAJOR = 0 means development before a stable API is reached.
  * Prerelease versions use an alphanumeric suffix without a separator.
    PEP 440 allows alpha versions (a0, a1, ...), beta versions (b0, b1, ...)
    and release candidates (rc0, rc1, ...).
  * Ongoing development on the GitHub main branch uses the version number
    x.y.0.dev1, where x.y is the *upcoming* release we ware working towards.
    We make no attempt to increment the "1" after "dev".
"""

# On the main branch, leave this at X.Y.0.dev1 ( = dev version of next release).
#
# To cut release X.Y.0, make a branch rX.Y and make commits on that branch
# to set "X.Y.0rc0", "X.Y.0rc1", ... (= release candidates) and eventually
# "X.Y.0" (= the release) as appropriate. We do not expect to do alpha/beta
# versions for an ordinary release.
#
# Patch releases X.Y.Z, Z > 0, incl. their release candidates, can be done for
# patches on branch rX.Y as needed.
#
# IMPORANT: Right after branching rX.Y, bump the main branch to X.(Y+1).0.dev1.
# (Submit a change to the Source of Truth, get it out on the main branch asap.)

__version__ = "0.2.0rc0"
