"""BAT stratum and delivered-regimen primitives for the REGAL v2 model.

REGAL's published BAT description contains two different quantities that must not
be collapsed into one set of weights:

* the planned randomization stratum (supportive care/hydroxyurea, HMA,
  venetoclax, or LDAC); and
* the regimen actually delivered to a patient, which may contain more than one
  component under the protocol's "and/or" wording.

This module represents their joint patient-level distribution.  Every pathway
has one stratum and one regimen.  A combination regimen can expose a patient to
multiple components, but it selects exactly one survival profile, so neither the
patient nor the outcome is counted twice.

The primary constant below reproduces approximately equal planned strata.  It is
a protocol-compatible proxy, not evidence of REGAL's realized regimen mix.  The
legacy, venetoclax-dominant, and bear constants are explicitly classified as a
comparison or stress tests.  The v1 low-venetoclax and bull presets are not carried
forward because work package 3 names only the venetoclax-dominant and bear
allocations.  Nothing in this module is a forecast for the ongoing trial, and it is
not imported by the legacy explorer.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from math import isclose, isfinite
from numbers import Integral
from types import MappingProxyType
from typing import Dict, Tuple

import numpy as np

from survival_models import (
    CureMixtureComponent,
    SurvivalScale,
    WeibullSurvival,
)


PROBABILITY_TOLERANCE = 1e-12


class BATStratum(str, Enum):
    """Planned BAT category recorded before treatment assignment."""

    SUPPORTIVE_CARE_HYDROXYUREA = "supportive_care_hydroxyurea"
    HMA = "hma"
    VENETOCLAX = "venetoclax"
    LDAC = "ldac"


class BATComponent(str, Enum):
    """Patient-level BAT exposures and available survival-profile keys."""

    OBSERVATION = "observation"
    HYDROXYUREA = "hydroxyurea"
    HMA = "hma"
    VENETOCLAX = "venetoclax"
    LDAC = "ldac"


class BATDesignRole(str, Enum):
    """How a BAT design may be used in v2 analyses."""

    PRIMARY = "primary"
    STRESS_TEST = "stress_test"
    LEGACY_COMPARISON = "legacy_comparison"


_STRATUM_COMPONENTS = {
    BATStratum.SUPPORTIVE_CARE_HYDROXYUREA: frozenset(
        (BATComponent.OBSERVATION, BATComponent.HYDROXYUREA)
    ),
    BATStratum.HMA: frozenset((BATComponent.HMA,)),
    BATStratum.VENETOCLAX: frozenset((BATComponent.VENETOCLAX,)),
    BATStratum.LDAC: frozenset((BATComponent.LDAC,)),
}


def _enum_value(value, enum_type, field_name):
    if isinstance(value, Enum) and not isinstance(value, enum_type):
        raise ValueError(
            f"{field_name} must be a {enum_type.__name__}, "
            f"not {type(value).__name__}"
        )
    try:
        return enum_type(value)
    except (TypeError, ValueError) as error:
        choices = ", ".join(member.value for member in enum_type)
        raise ValueError(f"{field_name} must be one of: {choices}") from error


def _validate_stratum_regimen(stratum, regimen):
    required = _STRATUM_COMPONENTS[stratum]
    if required.isdisjoint(regimen.components):
        raise ValueError(
            f"regimen {regimen.key!r} does not contain a component compatible "
            f"with stratum {stratum.value!r}"
        )


@dataclass(frozen=True)
class BATRegimen:
    """One delivered regimen and the single profile used for its outcome.

    ``components`` records all known exposures.  ``survival_component`` is the
    one component-library key used to generate the patient's event time.  For
    example, HMA + venetoclax records both exposures but uses the venetoclax
    profile. That profile is derived from VEN + azacitidine, making this explicit
    combination its best-supported mapping. Applying it to venetoclax with unknown
    co-therapy is a provisional, directionally BAT-favorable proxy that may
    overstate survival for venetoclax monotherapy.
    """

    key: str
    components: Tuple[BATComponent, ...]
    survival_component: BATComponent
    label: str = ""

    def __post_init__(self):
        if not isinstance(self.key, str) or not self.key.strip():
            raise ValueError("key must be a non-empty string")
        key = self.key.strip()
        try:
            raw_components = tuple(self.components)
        except TypeError as error:
            raise ValueError("components must contain only BATComponent values") from error
        supplied = tuple(
            _enum_value(component, BATComponent, "components")
            for component in raw_components
        )
        if not supplied:
            raise ValueError("components must contain at least one BAT component")
        if len(set(supplied)) != len(supplied):
            raise ValueError("components must not contain duplicates")
        components = tuple(component for component in BATComponent if component in supplied)
        survival_component = _enum_value(
            self.survival_component, BATComponent, "survival_component"
        )
        if survival_component not in components:
            raise ValueError("survival_component must also appear in components")
        if not isinstance(self.label, str):
            raise ValueError("label must be a string")
        label = self.label.strip() or key.replace("_", " ").title()
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "survival_component", survival_component)
        object.__setattr__(self, "label", label)


@dataclass(frozen=True)
class BATPathway:
    """One cell of the joint planned-stratum/delivered-regimen distribution.

    A zero probability explicitly retains an absent stratum in a non-primary
    design. It is never sampled and contributes no regimen, exposure, or profile
    mass.
    """

    stratum: BATStratum
    regimen: BATRegimen
    probability: float

    def __post_init__(self):
        stratum = _enum_value(self.stratum, BATStratum, "stratum")
        if not isinstance(self.regimen, BATRegimen):
            raise ValueError("regimen must be a BATRegimen")
        if isinstance(self.probability, bool):
            raise ValueError("probability must be finite and in [0, 1]")
        try:
            probability = float(self.probability)
        except (TypeError, ValueError) as error:
            raise ValueError("probability must be finite and in [0, 1]") from error
        if not isfinite(probability) or probability < 0.0 or probability > 1.0:
            raise ValueError("probability must be finite and in [0, 1]")
        _validate_stratum_regimen(stratum, self.regimen)
        object.__setattr__(self, "stratum", stratum)
        object.__setattr__(self, "probability", probability)


@dataclass(frozen=True)
class BATPatientAssignment:
    """The planned stratum and delivered regimen for one modeled patient."""

    stratum: BATStratum
    regimen: BATRegimen

    def __post_init__(self):
        stratum = _enum_value(self.stratum, BATStratum, "stratum")
        if not isinstance(self.regimen, BATRegimen):
            raise ValueError("regimen must be a BATRegimen")
        _validate_stratum_regimen(stratum, self.regimen)
        object.__setattr__(self, "stratum", stratum)


@dataclass(frozen=True)
class BATCohort:
    """Immutable patient assignments sampled from a :class:`BATDesign`."""

    assignments: Tuple[BATPatientAssignment, ...]

    def __post_init__(self):
        assignments = tuple(self.assignments)
        if not assignments:
            raise ValueError("assignments must contain at least one patient")
        if not all(isinstance(item, BATPatientAssignment) for item in assignments):
            raise ValueError("assignments must contain only BATPatientAssignment values")
        object.__setattr__(self, "assignments", assignments)

    @property
    def patient_count(self):
        return len(self.assignments)

    def stratum_counts(self):
        counts = {stratum: 0 for stratum in BATStratum}
        for assignment in self.assignments:
            counts[assignment.stratum] += 1
        return counts

    def regimen_counts(self):
        counts: Dict[BATRegimen, int] = {}
        for assignment in self.assignments:
            regimen = assignment.regimen
            counts[regimen] = counts.get(regimen, 0) + 1
        return counts

    def survival_component_counts(self):
        """Count each patient once under the profile that generates its outcome."""

        counts = {component: 0 for component in BATComponent}
        for assignment in self.assignments:
            counts[assignment.regimen.survival_component] += 1
        return counts

    def component_exposure_counts(self):
        """Count component exposures; combinations can make the total exceed N."""

        counts = {component: 0 for component in BATComponent}
        for assignment in self.assignments:
            for component in assignment.regimen.components:
                counts[component] += 1
        return counts


@dataclass(frozen=True)
class BATDesign:
    """A joint distribution over planned BAT strata and delivered regimens.

    All designs name all four protocol strata. Primary designs must give every
    stratum positive mass; comparison and stress designs may use explicit
    zero-probability pathways to represent an absent stratum. Pathway totals are
    accepted within ``PROBABILITY_TOLERANCE`` and retain their supplied floats.
    """

    name: str
    role: BATDesignRole
    pathways: Tuple[BATPathway, ...]
    description: str = ""

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        name = self.name.strip()
        role = _enum_value(self.role, BATDesignRole, "role")
        pathways = tuple(self.pathways)
        if not pathways or not all(isinstance(item, BATPathway) for item in pathways):
            raise ValueError("pathways must contain BATPathway values")
        if {pathway.stratum for pathway in pathways} != set(BATStratum):
            raise ValueError("pathways must represent all four planned BAT strata")
        regimens = {}
        for pathway in pathways:
            prior = regimens.setdefault(pathway.regimen.key, pathway.regimen)
            if prior != pathway.regimen:
                raise ValueError("each regimen key must identify one regimen definition")
        cells = [(pathway.stratum, pathway.regimen.key) for pathway in pathways]
        if len(set(cells)) != len(cells):
            raise ValueError("each stratum/regimen pathway must be unique")
        total = sum(pathway.probability for pathway in pathways)
        if not isclose(
            total,
            1.0,
            rel_tol=PROBABILITY_TOLERANCE,
            abs_tol=PROBABILITY_TOLERANCE,
        ):
            raise ValueError("pathway probabilities must sum to 1")
        stratum_probabilities = {stratum: 0.0 for stratum in BATStratum}
        for pathway in pathways:
            stratum_probabilities[pathway.stratum] += pathway.probability
        if role is BATDesignRole.PRIMARY and any(
            probability <= 0.0 for probability in stratum_probabilities.values()
        ):
            raise ValueError("primary designs require positive mass in all four strata")
        if not isinstance(self.description, str):
            raise ValueError("description must be a string")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "pathways", pathways)
        object.__setattr__(self, "description", self.description.strip())

    @property
    def stratum_probabilities(self):
        probabilities = {stratum: 0.0 for stratum in BATStratum}
        for pathway in self.pathways:
            probabilities[pathway.stratum] += pathway.probability
        return probabilities

    @property
    def regimen_probabilities(self):
        """Patient shares by regimen, summing within ``PROBABILITY_TOLERANCE``."""

        probabilities: Dict[BATRegimen, float] = {}
        for pathway in self.pathways:
            probabilities[pathway.regimen] = (
                probabilities.get(pathway.regimen, 0.0) + pathway.probability
            )
        return probabilities

    @property
    def survival_component_probabilities(self):
        """Outcome-profile shares, summing within ``PROBABILITY_TOLERANCE``."""

        probabilities = {component: 0.0 for component in BATComponent}
        for pathway in self.pathways:
            probabilities[pathway.regimen.survival_component] += pathway.probability
        return probabilities

    @property
    def component_exposure_probabilities(self):
        """Exposure shares, whose total may exceed one for combination regimens."""

        probabilities = {component: 0.0 for component in BATComponent}
        for pathway in self.pathways:
            for component in pathway.regimen.components:
                probabilities[component] += pathway.probability
        return probabilities

    def validate_library(self, library=None):
        """Validate that every positive-mass outcome profile can be resolved.

        Zero-mass pathways explicitly represent absent strata in non-primary
        designs and therefore do not require an otherwise-unused survival profile.
        Extra library entries are permitted.
        """

        if library is None:
            library = DEFAULT_COMPONENT_LIBRARY
        if not isinstance(library, Mapping):
            raise ValueError("component library must be a mapping")
        required = {
            pathway.regimen.survival_component
            for pathway in self.pathways
            if pathway.probability > 0.0
        }
        missing = sorted(
            (component for component in required if component not in library),
            key=lambda component: component.value,
        )
        if missing:
            names = ", ".join(component.value for component in missing)
            raise ValueError(f"component library is missing survival profiles: {names}")
        invalid = sorted(
            (
                component
                for component in required
                if not isinstance(library[component], CureMixtureComponent)
            ),
            key=lambda component: component.value,
        )
        if invalid:
            names = ", ".join(component.value for component in invalid)
            raise ValueError(
                f"component library contains invalid survival profiles: {names}"
            )
        return library

    def sample(self, size, rng):
        """Sample patient assignments with one categorical draw per patient."""

        if isinstance(size, bool) or not isinstance(size, Integral) or size < 1:
            raise ValueError("size must be a positive integer")
        size = int(size)
        draws = np.asarray(rng.random(size), dtype=float)
        if draws.shape != (size,) or np.any(~np.isfinite(draws)):
            raise ValueError("rng.random(size) must return finite one-dimensional draws")
        if np.any(draws < 0.0) or np.any(draws >= 1.0):
            raise ValueError("rng.random(size) draws must be in [0, 1)")
        cumulative = np.cumsum([pathway.probability for pathway in self.pathways])
        cumulative /= cumulative[-1]
        cumulative[-1] = 1.0
        indices = np.searchsorted(cumulative, draws, side="right")
        assignments = tuple(
            BATPatientAssignment(
                stratum=self.pathways[index].stratum,
                regimen=self.pathways[index].regimen,
            )
            for index in indices
        )
        return BATCohort(assignments)


OBSERVATION_REGIMEN = BATRegimen(
    "observation",
    (BATComponent.OBSERVATION,),
    BATComponent.OBSERVATION,
    "Observation / best supportive care",
)
HYDROXYUREA_REGIMEN = BATRegimen(
    "hydroxyurea",
    (BATComponent.HYDROXYUREA,),
    BATComponent.HYDROXYUREA,
    "Hydroxyurea",
)
HMA_REGIMEN = BATRegimen(
    "hma",
    (BATComponent.HMA,),
    BATComponent.HMA,
    "HMA monotherapy",
)
VENETOCLAX_UNSPECIFIED_REGIMEN = BATRegimen(
    "venetoclax_unspecified",
    (BATComponent.VENETOCLAX,),
    BATComponent.VENETOCLAX,
    "Venetoclax-based (co-therapy unspecified)",
)
LDAC_REGIMEN = BATRegimen(
    "ldac",
    (BATComponent.LDAC,),
    BATComponent.LDAC,
    "Low-dose cytarabine",
)
HMA_VENETOCLAX_REGIMEN = BATRegimen(
    "hma_venetoclax",
    (BATComponent.HMA, BATComponent.VENETOCLAX),
    BATComponent.VENETOCLAX,
    "HMA + venetoclax",
)
LDAC_VENETOCLAX_REGIMEN = BATRegimen(
    "ldac_venetoclax",
    (BATComponent.LDAC, BATComponent.VENETOCLAX),
    BATComponent.VENETOCLAX,
    "LDAC + venetoclax",
)


def _proxy_pathways(observation, hydroxyurea, hma, venetoclax, ldac):
    """Build the single-profile proxy used by committed comparison designs."""

    return (
        BATPathway(
            BATStratum.SUPPORTIVE_CARE_HYDROXYUREA,
            OBSERVATION_REGIMEN,
            observation,
        ),
        BATPathway(
            BATStratum.SUPPORTIVE_CARE_HYDROXYUREA,
            HYDROXYUREA_REGIMEN,
            hydroxyurea,
        ),
        BATPathway(BATStratum.HMA, HMA_REGIMEN, hma),
        BATPathway(
            BATStratum.VENETOCLAX,
            VENETOCLAX_UNSPECIFIED_REGIMEN,
            venetoclax,
        ),
        BATPathway(BATStratum.LDAC, LDAC_REGIMEN, ldac),
    )


PRIMARY_EQUAL_STRATA = BATDesign(
    name="protocol_equal_strata",
    role=BATDesignRole.PRIMARY,
    pathways=_proxy_pathways(0.25 * 27.0 / 35.0, 0.25 * 8.0 / 35.0, 0.25, 0.25, 0.25),
    description=(
        "Approximately 25% per planned stratum. Single-profile regimens are an "
        "explicit proxy until realized combination evidence is available."
    ),
)

LEGACY_COMPONENT_MIX = BATDesign(
    name="legacy_component_mix",
    role=BATDesignRole.LEGACY_COMPARISON,
    pathways=_proxy_pathways(0.27, 0.08, 0.22, 0.35, 0.08),
    description="Current v1 component weights, retained only as a legacy comparison.",
)

VENETOCLAX_DOMINANT_STRESS = BATDesign(
    name="venetoclax_dominant",
    role=BATDesignRole.STRESS_TEST,
    pathways=_proxy_pathways(0.08, 0.04, 0.18, 0.60, 0.10),
    description="US-heavy component-weight allocation stress test.",
)

BEAR_STRONG_BAT_STRESS = BATDesign(
    name="bear_strong_bat",
    role=BATDesignRole.STRESS_TEST,
    pathways=_proxy_pathways(0.05, 0.03, 0.12, 0.70, 0.10),
    description=(
        "Strong-BAT allocation stress test. The separate legacy 25% venetoclax "
        "cure override belongs to survival-parameter sensitivity, not allocation."
    ),
)


def _make_default_component_library():
    """Return the current literature inputs on their documented OS scale."""

    return {
        BATComponent.OBSERVATION: CureMixtureComponent(
            "Observation / best supportive care",
            WeibullSurvival(6.0, 1.1),
            0.03,
            SurvivalScale.OVERALL,
        ),
        BATComponent.HYDROXYUREA: CureMixtureComponent(
            "Hydroxyurea",
            WeibullSurvival(5.0, 1.1),
            0.02,
            SurvivalScale.OVERALL,
        ),
        BATComponent.HMA: CureMixtureComponent(
            "HMA",
            WeibullSurvival(12.0, 1.0),
            0.10,
            SurvivalScale.OVERALL,
        ),
        BATComponent.VENETOCLAX: CureMixtureComponent(
            "Venetoclax-based",
            WeibullSurvival(12.0, 0.78),
            0.15,
            SurvivalScale.OVERALL,
        ),
        BATComponent.LDAC: CureMixtureComponent(
            "LDAC",
            WeibullSurvival(7.0, 1.1),
            0.08,
            SurvivalScale.OVERALL,
        ),
    }


DEFAULT_COMPONENT_LIBRARY: Mapping[BATComponent, CureMixtureComponent] = (
    MappingProxyType(_make_default_component_library())
)


def default_component_library():
    """Return a fresh mapping of the default, immutable survival components."""

    return dict(DEFAULT_COMPONENT_LIBRARY)


def _component_library_with_cure(library, component, cure_fraction):
    """Copy a component library with one explicit cure-fraction sensitivity."""

    component = _enum_value(component, BATComponent, "component")
    if not isinstance(library, Mapping):
        raise ValueError("component library must be a mapping")
    try:
        source = library[component]
    except KeyError as error:
        raise ValueError(
            f"component library is missing survival profile: {component.value}"
        ) from error
    if not isinstance(source, CureMixtureComponent):
        raise ValueError(
            f"component library contains invalid survival profile: {component.value}"
        )
    result = dict(library)
    result[component] = CureMixtureComponent(
        name=source.name,
        uncured=source.uncured,
        cure_fraction=cure_fraction,
        survival_scale=source.survival_scale,
    )
    return result


BEAR_STRONG_BAT_COMPONENT_LIBRARY: Mapping[BATComponent, CureMixtureComponent] = (
    MappingProxyType(
        _component_library_with_cure(
            DEFAULT_COMPONENT_LIBRARY,
            BATComponent.VENETOCLAX,
            0.25,
        )
    )
)


def component_for(assignment, library=DEFAULT_COMPONENT_LIBRARY):
    """Return the single survival component that generates a patient's outcome."""

    if not isinstance(assignment, BATPatientAssignment):
        raise ValueError("assignment must be a BATPatientAssignment")
    if not isinstance(library, Mapping):
        raise ValueError("component library must be a mapping")
    key = assignment.regimen.survival_component
    try:
        component = library[key]
    except KeyError as error:
        raise ValueError(
            f"component library is missing survival profile: {key.value}"
        ) from error
    if not isinstance(component, CureMixtureComponent):
        raise ValueError(
            f"component library contains invalid survival profile: {key.value}"
        )
    return component


__all__ = [
    "BATComponent",
    "BATCohort",
    "BATDesign",
    "BATDesignRole",
    "BATPathway",
    "BATPatientAssignment",
    "BATRegimen",
    "BATStratum",
    "BEAR_STRONG_BAT_COMPONENT_LIBRARY",
    "BEAR_STRONG_BAT_STRESS",
    "DEFAULT_COMPONENT_LIBRARY",
    "HMA_REGIMEN",
    "HMA_VENETOCLAX_REGIMEN",
    "HYDROXYUREA_REGIMEN",
    "LDAC_REGIMEN",
    "LDAC_VENETOCLAX_REGIMEN",
    "LEGACY_COMPONENT_MIX",
    "OBSERVATION_REGIMEN",
    "PRIMARY_EQUAL_STRATA",
    "PROBABILITY_TOLERANCE",
    "VENETOCLAX_DOMINANT_STRESS",
    "VENETOCLAX_UNSPECIFIED_REGIMEN",
    "component_for",
    "default_component_library",
]
