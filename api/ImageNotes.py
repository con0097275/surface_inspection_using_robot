from pydantic import BaseModel


class ImageNote(BaseModel):
    building: str
    image: str