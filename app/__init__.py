# PatchContext app package

# Monkeypatch pydantic v1 BaseModel.__setstate__ to support unpickling v2 models 
# (which might be saved in FAISS docstore during pickling and fail on missing '__fields_set__')
try:
    import pydantic.v1.main
    orig_setstate = pydantic.v1.main.BaseModel.__setstate__

    def _fixed_setstate(self, state):
        if '__fields_set__' not in state:
            if '__pydantic_fields_set__' in state:
                state['__fields_set__'] = state['__pydantic_fields_set__']
            else:
                state['__fields_set__'] = set()
        return orig_setstate(self, state)

    pydantic.v1.main.BaseModel.__setstate__ = _fixed_setstate
except ImportError:
    pass
