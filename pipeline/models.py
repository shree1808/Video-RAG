from pydantic import BaseModel, Field

class MetadataExtraction_Frame(BaseModel):
    emotion: str = Field(
        description="The predominant emotional state or expression conveyed by the main subject(s) in the frame (e.g., happy, angry, calm)."
    )
    action: str = Field(
        description="The primary activity, movement, or behavior being performed by the subject(s) or object(s) in the frame."
    )
    caption: str = Field(
        description="A concise, well-structured sentence that summarizes the key elements of the frame, including the main subjects, objects, actions, and notable physical details."
    )
    description : str = Field( 
        description= "Brief description of the frame covering all the figures, statistics/numbers, and overall summary"
    )
