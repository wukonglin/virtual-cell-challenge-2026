"""Pure NumPy GenePT feature construction: no I/O, response fitting or training.

Availability is embedding coverage, never a null-perturbation flag. GO remains
separate and unchanged. Vector sharing never establishes biological aliases.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping
import numpy as np

SCHEMA = "genept-conditioning-v11"
DIMENSION, MAX_TARGETS = 1536, 20000


def require(value, message):
    if not value:
        raise ValueError(message)


def _names(values, label, *, empty=False):
    require(not isinstance(values,(str,bytes)), label+" must be a sequence")
    try:
        result = tuple(values)
    except TypeError:
        raise ValueError(label+" must be a sequence") from None
    require((empty or result) and len(result) <= MAX_TARGETS
        and all(type(v) in (str,np.str_) and v and v == v.strip() and len(v) <= 256
            and all(ch.isprintable() for ch in v) for v in result)
        and len(set(result)) == len(result), "Unique bounded exact "+label+" required")
    return tuple(str(v) for v in result)


def _json_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,
        separators=(",",":"),allow_nan=False).encode()).hexdigest()


def _readonly(array):
    result = np.ascontiguousarray(array).copy()
    result.flags.writeable = False
    return result


def _hex(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


@dataclass(frozen=True)
class GenePTFeatures:
    target_ids: tuple[str,...]
    embeddings: np.ndarray
    present: np.ndarray
    source_vector_object_id: np.ndarray


def prepare_features(target_ids, embeddings, present, *, source_vector_object_id=None,
                     expected_dimension=DIMENSION):
    """Validate and privately copy named raw numeric features into read-only arrays."""
    ids = _names(target_ids,"target names")
    require(type(expected_dimension) is int and 1 <= expected_dimension <= 4096,"Bounded embedding dimension required")
    values,available = np.asarray(embeddings),np.asarray(present)
    require(values.shape == (len(ids),expected_dimension) and values.dtype.kind == "f"
        and values.dtype.itemsize in (4,8) and np.isfinite(values).all(),
        "Finite float32/float64 named embedding matrix required")
    require(available.shape == (len(ids),) and available.dtype.kind in "biu"
        and np.isin(available,(0,1)).all(),"Binary aligned availability required")
    require(np.max(np.abs(values)) <= np.finfo(np.float32).max,"Raw embedding float32 range exceeded")
    available = available.astype(np.uint8)
    require(np.all(values[available == 0] == 0),"Unavailable source rows must be explicit zero vectors")
    require(np.all(np.any(values[available == 1] != 0,axis=1)),"Available source rows cannot be zero vectors")
    if source_vector_object_id is None:
        references = np.full(len(ids),-1,dtype=np.int64)
    else:
        references = np.asarray(source_vector_object_id)
        require(references.shape == (len(ids),) and references.dtype.kind in "iu"
            and np.all(references >= -1) and np.all(references <= np.iinfo(np.int32).max),
            "Aligned bounded source object IDs required")
        references = references.astype(np.int64)
        require(np.all(references[available == 0] == -1),"Missing embeddings cannot reference a source object")
        first = {}
        for i,ref in enumerate(references.tolist()):
            if ref < 0:
                continue
            if ref in first:
                require(np.array_equal(values[i],values[first[ref]]),"Shared source object IDs have conflicting numeric content")
            else:
                first[ref] = i
    return GenePTFeatures(ids,_readonly(values),_readonly(available),_readonly(references))


def _validate_table(table):
    require(isinstance(table,GenePTFeatures),"Prepared named GenePT table required")
    shape = np.shape(table.embeddings)
    return prepare_features(table.target_ids,table.embeddings,table.present,
        source_vector_object_id=table.source_vector_object_id,
        expected_dimension=shape[1] if len(shape) == 2 else 0)


def _vector_sha(values):
    vector = np.asarray(values,dtype="<f8").copy()
    vector[vector == 0] = 0.0  # Signed zeros are numerically equal; no approximate matching.
    return hashlib.sha256(vector.tobytes()).hexdigest()


def _table_sha(table):
    digest = hashlib.sha256(_feature_sha(table,table.target_ids).encode("ascii"))
    digest.update(np.asarray(table.source_vector_object_id,dtype="<i8").tobytes())
    return digest.hexdigest()


def quarantine_duplicate_features(table, *, raw_artifact_sha256=None):
    """Mask every member of distinct-name exact duplicate groups, without relabeling.

    The optional artifact hash is supplied/authenticated by the caller; this
    module cannot read or authenticate files. Response membership is unchanged.
    """
    table = _validate_table(table)
    require(raw_artifact_sha256 is None or _hex(raw_artifact_sha256),"Valid optional raw artifact SHA256 required")
    groups = {}
    for i,name in enumerate(table.target_ids):
        if table.present[i]:
            groups.setdefault(_vector_sha(table.embeddings[i]),[]).append(i)
    values,present,objects = table.embeddings.copy(),table.present.copy(),table.source_vector_object_id.copy()
    records = []
    for sha,indices in sorted(groups.items()):
        if len(indices) < 2:
            continue
        require(all(np.array_equal(table.embeddings[indices[0]],table.embeddings[i]) for i in indices),
            "Exact duplicate hash collision")
        names = [table.target_ids[i] for i in indices]
        records.append({"target_names":sorted(names),"canonical_binary64_sha256":sha,
            "source_vector_object_ids":{table.target_ids[i]:int(table.source_vector_object_id[i]) for i in indices},
            "action":"all distinct-name members unavailable; no member chosen as representative"})
        values[indices] = 0
        present[indices] = 0
        objects[indices] = -1
    filtered = prepare_features(table.target_ids,values,present,source_vector_object_id=objects,
        expected_dimension=values.shape[1])
    quarantined = sorted(name for group in records for name in group["target_names"])
    return filtered,{"schema":SCHEMA,"operation":"conservative_exact_duplicate_feature_quarantine",
        "input_table_sha256":_table_sha(table),"output_table_sha256":_table_sha(filtered),
        "caller_supplied_raw_artifact_sha256":raw_artifact_sha256,
        "artifact_hash_authenticated_by_this_module":False,"duplicate_groups":records,
        "quarantined_target_names":quarantined,"quarantined_count":len(quarantined),
        "available_before":int(table.present.sum()),"available_after":int(filtered.present.sum()),
        "reason":"ambiguous distinct-name identical pretrained representations; conservative feature admission only",
        "original_table_modified":False,"labels_changed":False,"GO_features_modified":False,
        "response_roles_changed":False,"biological_alias_equivalence_established":False,
        "model_training_performed":False,"missing_vectors_excluded_from_duplicate_groups":True,
        "numeric_content_rule":"exact available values, little-endian binary64, signed zeros canonicalized"}


def _roles(table, fit_targets, held_targets, official_targets=()):
    fit = tuple(sorted(_names(fit_targets,"fitting target names")))
    held = tuple(sorted(_names(held_targets,"held target names",empty=True)))
    official = tuple(sorted(_names(official_targets,"official target names",empty=True)))
    require(not set(fit)&set(held),"Fitting and held target identities overlap")
    require(not set(fit)&set(official),"Official target identities cannot fit a transform")
    require(set(fit)|set(held)|set(official) <= set(table.target_ids),"Every registered role name must exist in the named table")
    return fit,held,official


def _feature_sha(table, names):
    lookup = {name:i for i,name in enumerate(table.target_ids)}
    indices = [lookup[name] for name in names]
    digest = hashlib.sha256(_json_sha(list(names)).encode("ascii"))
    digest.update(np.asarray(table.present[indices],dtype=np.uint8).tobytes())
    digest.update(np.asarray(table.embeddings[indices],dtype="<f8").tobytes())
    return digest.hexdigest()


_SEMANTICS = {
    "rule":"coordinate mean then one scalar RMS over available fitting rows and coordinates",
    "statistics_dtype":"float64","output_dtype":"float32",
    "missing_policy":"zero vector with present=0; availability is not a null-perturbation flag",
    "fitting_scope":"available source fitting-target embeddings only",
    "responses_used":False,"model_training_performed":False}


@dataclass(frozen=True)
class GenePTTransform:
    source_id: str
    dimension: int
    fit_targets: tuple[str,...]
    available_fit_targets: tuple[str,...]
    fit_feature_sha256: str
    mean: tuple[float,...]
    scalar_rms: float

    def to_dict(self):
        value = {"schema":SCHEMA,"source_id":self.source_id,"dimension":self.dimension,
            "fit_targets":list(self.fit_targets),"fit_targets_sha256":_json_sha(list(self.fit_targets)),
            "available_fit_targets":list(self.available_fit_targets),
            "available_fit_targets_sha256":_json_sha(list(self.available_fit_targets)),
            "fit_feature_sha256":self.fit_feature_sha256,"mean":list(self.mean),"scalar_rms":self.scalar_rms,
            "available_fit_rows":len(self.available_fit_targets),
            "rms_coordinate_count":len(self.available_fit_targets)*self.dimension,**_SEMANTICS}
        _validate_transform_record(value)
        return value

    @classmethod
    def from_dict(cls, record):
        _validate_transform_record(record)
        return cls(record["source_id"],record["dimension"],tuple(record["fit_targets"]),
            tuple(record["available_fit_targets"]),record["fit_feature_sha256"],
            tuple(float(v) for v in record["mean"]),float(record["scalar_rms"]))


def _validate_transform_record(record):
    keys = {"schema","source_id","dimension","fit_targets","fit_targets_sha256","available_fit_targets",
        "available_fit_targets_sha256","fit_feature_sha256","mean","scalar_rms","available_fit_rows",
        "rms_coordinate_count",*_SEMANTICS}
    require(isinstance(record,dict) and set(record) == keys,"Exact transform record fields required")
    require(record["schema"] == SCHEMA,"Transform schema differs")
    _names([record["source_id"]],"source ID")
    dimension = record["dimension"]
    require(type(dimension) is int and 1 <= dimension <= 4096,"Transform dimension bound")
    fit = _names(record["fit_targets"],"fitting target names")
    present = _names(record["available_fit_targets"],"available fitting target names")
    require(fit == tuple(sorted(fit)) and present == tuple(sorted(present))
        and set(present) <= set(fit) and len(present) >= 2,"Sorted available fitting subset required")
    require(record["fit_targets_sha256"] == _json_sha(list(fit))
        and record["available_fit_targets_sha256"] == _json_sha(list(present))
        and _hex(record["fit_feature_sha256"]),"Transform fitting identity hash differs")
    mean,rms = record["mean"],record["scalar_rms"]
    require(isinstance(mean,list) and len(mean) == dimension
        and all(type(v) in (int,float) and math.isfinite(v) for v in mean),"Finite coordinate mean required")
    require(type(rms) in (int,float) and math.isfinite(rms) and rms > 0,"Positive finite scalar RMS required")
    require(type(record["available_fit_rows"]) is int and record["available_fit_rows"] == len(present)
        and type(record["rms_coordinate_count"]) is int and record["rms_coordinate_count"] == len(present)*dimension,
        "Transform sample counts differ")
    require(all(type(record[k]) is type(v) and record[k] == v for k,v in _SEMANTICS.items()),"Transform semantics differ")


def fit_transform(table, *, source_id, fit_targets, held_targets=(), official_targets=()):
    """Fit only embedding statistics, not expression responses or a predictive model."""
    table = _validate_table(table)
    _names([source_id],"source ID")
    fit,_,_ = _roles(table,fit_targets,held_targets,official_targets)
    lookup = {name:i for i,name in enumerate(table.target_ids)}
    available = tuple(name for name in fit if table.present[lookup[name]])
    require(len(available) >= 2,"At least two available fitting embeddings required")
    values = table.embeddings[[lookup[name] for name in available]].astype(np.float64)
    mean = values.mean(axis=0,dtype=np.float64)
    centered = values-mean
    rms = float(np.sqrt(np.mean(centered*centered,dtype=np.float64)))
    require(np.isfinite(mean).all() and math.isfinite(rms) and rms > 0,"Zero or nonfinite source fitting scale")
    result = GenePTTransform(source_id,values.shape[1],fit,available,_feature_sha(table,fit),
        tuple(float(v) for v in mean),rms)
    result.to_dict()
    return result


@dataclass(frozen=True)
class TransformedGenePT:
    target_ids: tuple[str,...]
    embeddings: np.ndarray
    present: np.ndarray
    source_id: str
    transform_sha256: str
    fitting_target_ids: tuple[str,...]


def transform_features(table, transform, *, target_ids=None):
    """Apply frozen statistics to arbitrary named rows without refitting."""
    table = _validate_table(table)
    require(isinstance(transform,GenePTTransform),"Frozen GenePT transform required")
    record = transform.to_dict()
    require(table.embeddings.shape[1] == transform.dimension,"New features have a different dimension")
    names = table.target_ids if target_ids is None else _names(target_ids,"selected target names")
    lookup = {name:i for i,name in enumerate(table.target_ids)}
    require(set(names) <= set(lookup),"Unknown selected target name")
    positions = np.asarray([lookup[name] for name in names],dtype=np.int64)
    present = table.present[positions]
    values = np.zeros((len(names),transform.dimension),dtype=np.float32)
    included = np.flatnonzero(present)
    normalized = (table.embeddings[positions[included]].astype(np.float64)
        -np.asarray(transform.mean,dtype=np.float64))/transform.scalar_rms
    require(np.isfinite(normalized).all() and (np.abs(normalized) <= np.finfo(np.float32).max).all(),
        "Transformed embedding exceeds float32 range")
    values[included] = normalized
    require(np.isfinite(values).all() and np.all(values[present == 0] == 0),"Invalid transformed missing rows")
    return TransformedGenePT(tuple(names),_readonly(values),_readonly(present),transform.source_id,_json_sha(record),
        transform.fit_targets)


@dataclass(frozen=True)
class GenePTArm:
    name: str
    target_ids: tuple[str,...]
    embeddings: np.ndarray
    present: np.ndarray
    donor_target_ids: tuple[str | None,...]
    report: dict


def build_arms(features, *, fit_targets, held_targets, seed):
    require(isinstance(features,TransformedGenePT),"Transformed named features required")
    ids = _names(features.target_ids,"target names")
    values,present = np.asarray(features.embeddings),np.asarray(features.present)
    require(values.ndim == 2 and values.shape[0] == len(ids) and 1 <= values.shape[1] <= 4096
        and values.dtype == np.float32 and np.isfinite(values).all(),"Finite transformed float32 features required")
    require(present.shape == (len(ids),) and present.dtype == np.uint8 and np.isin(present,(0,1)).all()
        and np.all(values[present == 0] == 0),"Aligned unchanged availability required")
    require(_hex(features.transform_sha256),"Frozen transform hash required")
    _names([features.source_id],"source ID")
    fit = tuple(sorted(_names(fit_targets,"fitting names")))
    held = tuple(sorted(_names(held_targets,"held names",empty=True)))
    require(not set(fit)&set(held) and set(fit)|set(held) == set(ids),
        "Fit and held identities must exactly partition arm targets")
    require(fit == _names(features.fitting_target_ids,"transform fitting target names"),
        "Arm fitting identities differ from frozen statistic-fitting identities")
    require(type(seed) is int and 0 <= seed < 2**64,"Independent uint64 shuffle seed required")
    lookup = {name:i for i,name in enumerate(ids)}
    donors = list(ids)
    strata,singletons = [],[]
    for role,names in (("fit",fit),("held",held)):
        for availability in (0,1):
            members = [name for name in names if int(present[lookup[name]]) == availability]
            if len(members) >= 2:
                derived = int(_json_sha([SCHEMA,seed,features.source_id,role,availability]),16)
                rng = np.random.default_rng(derived)
                order = rng.permutation(len(members))
                for recipient,donor in zip(order,np.roll(order,1)):
                    donors[lookup[members[int(recipient)]]] = members[int(donor)]
            elif len(members) == 1:
                singletons.append({"role":role,"present":availability,"target":members[0]})
            strata.append({"role":role,"present":availability,"size":len(members),
                "deranged":len(members) >= 2,"unchanged_singleton":len(members) == 1})
    shuffled = values[[lookup[name] for name in donors]]
    require(np.array_equal(present,present[[lookup[name] for name in donors]]),"Shuffle changed availability")
    common = {"schema":SCHEMA,"source_id":features.source_id,"transform_sha256":features.transform_sha256,
        "fit_targets_sha256":_json_sha(list(fit)),"held_targets_sha256":_json_sha(list(held)),
        "seed":seed,"availability_preserved":True,"availability_is_null_perturbation_flag":False,
        "GO_features_modified":False,"model_training_performed":False,
        "shuffle_rule":"independent_role_availability_rng_cyclic_derangement"}
    return {name:GenePTArm(name,ids,_readonly(matrix),_readonly(present),tuple(mapping),
        {**common,"arm":name,"strata":strata if name == "shuffled" else [],
         "unchanged_singletons":singletons if name == "shuffled" else [],
         "numerically_unchanged_rows":int(np.count_nonzero(np.all(matrix == values,axis=1)))})
        for name,matrix,mapping in (("correct",values,ids),
            ("masked",np.zeros_like(values),[None]*len(ids)),("shuffled",shuffled,donors))}


def audit_sharing(table, roles: Mapping[str,tuple[str,...] | list[str]]):
    """Audit cross-role object and numeric sharing; never revise biological identities."""
    table = _validate_table(table)
    require(isinstance(roles,Mapping) and 2 <= len(roles) <= 32,"At least two named roles required")
    lookup = {name:i for i,name in enumerate(table.target_ids)}
    normalized = {}
    for role,names in roles.items():
        _names([role],"role label")
        normalized[role] = tuple(sorted(_names(names,"role targets",empty=True)))
        require(set(normalized[role]) <= set(lookup),"Unknown audit role target")
    memberships = {name:sorted(role for role,names in normalized.items() if name in names)
        for name in sorted(set().union(*map(set,normalized.values())))}
    identities = [{"target":name,"roles":assigned} for name,assigned in memberships.items() if len(assigned) > 1]
    content,objects = {},{}
    for name in memberships:
        index = lookup[name]
        if not table.present[index]:
            continue
        digest = _vector_sha(table.embeddings[index])
        content.setdefault(digest,[]).append(name)
        reference = int(table.source_vector_object_id[index])
        if reference >= 0:
            objects.setdefault(reference,[]).append(name)
    def clusters(groups, field):
        result = []
        for key,names in sorted(groups.items(),key=lambda pair:str(pair[0])):
            assigned = sorted(set().union(*(set(memberships[name]) for name in names)))
            if len(names) >= 2 and len(assigned) >= 2:
                result.append({field:key,"target_names":sorted(names),"roles":assigned,
                    "memberships":{name:memberships[name] for name in sorted(names)}})
        return result
    shared_content = clusters(content,"canonical_binary64_sha256")
    shared_objects = clusters(objects,"source_vector_object_id")
    review = bool(identities or shared_content or shared_objects)
    return {"schema":SCHEMA,"roles":{role:list(names) for role,names in normalized.items()},
        "identity_overlap":identities,"exact_numeric_content_sharing":shared_content,
        "source_object_sharing":shared_objects,"requires_training_admission_review":review,
        "warnings":["Cross-role identity/vector sharing requires explicit training-admission review"] if review else [],
        "biological_alias_equivalence_established":False,"labels_changed":False,"roles_changed":False,
        "training_authorized":False,"missing_vectors_excluded_from_sharing":True,
        "numeric_content_rule":"exact available values, little-endian binary64, signed zeros canonicalized"}
