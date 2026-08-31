from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from credit_ai.api.schemas import FeaturePayload, ScoreResponse, StatusResponse
from credit_ai.features.schema import features_from_mapping
from credit_ai.service import CreditService

router = APIRouter(prefix="/v1")


def _service(request: Request) -> CreditService:
    return request.app.state.service


@router.get("/status", response_model=StatusResponse)
def status(request: Request) -> StatusResponse:
    service = _service(request)
    return StatusResponse(
        ok=True,
        mode=service.settings.app_mode,
        db_configured=service.settings.db_configured,
        using_database=service.settings.use_database,
        model_loaded=service.scorer is not None,
        secrets_source=service.settings.db_secrets_from,
        gcp_project_id=service.settings.gcp_project_id,
    )


@router.post("/score", response_model=ScoreResponse)
def score_features(payload: FeaturePayload, request: Request) -> ScoreResponse:
    service = _service(request)
    try:
        bundle = service.score_features(features_from_mapping(payload.model_dump()))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ScoreResponse.model_validate(bundle.as_dict())


@router.post("/score/member/{member_id}", response_model=ScoreResponse)
def score_member(member_id: int, request: Request) -> ScoreResponse:
    service = _service(request)
    try:
        bundle = service.score_member(member_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ScoreResponse.model_validate(bundle.as_dict())


@router.post("/score/class", response_model=list[ScoreResponse])
def score_class(request: Request) -> list[ScoreResponse]:
    service = _service(request)
    try:
        bundles = service.score_class()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return [ScoreResponse.model_validate(item.as_dict()) for item in bundles]


@router.get("/class")
def list_class(request: Request) -> list[dict]:
    service = _service(request)
    try:
        return service.list_class()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/report/{member_id}", response_model=ScoreResponse)
def get_report(member_id: int, request: Request) -> ScoreResponse:
    return score_member(member_id, request)
