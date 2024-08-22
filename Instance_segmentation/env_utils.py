ROOM_LIST = [
    "bathroom", 
    "bedroom",
    "kitchen",
    "livingroom", 
]

SURFACE_LIST = [
    "coffeetable",
    "desk",
    "kitchentable",
    "sofa",
    "kitchencounter",
    "bathroomcounter"
]

CONTAINER_LIST = [
    "kitchencabinet",
    "cabinet",
    "bathroomcabinet",
    "dishwasher",
    "fridge",
    "microwave",
    "stove",
]

NEW_CONTAINER_LIST = [
    "1st kitchencabinet from left to right",
    "2nd kitchencabinet from left to right",
    "3rd kitchencabinet from left to right",
    "4th kitchencabinet from left to right",
    "5th kitchencabinet from left to right",
    "6th kitchencabinet from left to right",
    "7th kitchencabinet from left to right",
    "8th kitchencabinet from left to right",
    "cabinet",
    "bathroomcabinet",
    "dishwasher",
    "fridge",
    "microwave",
    "stove",
]

OBJECT_LIST = ["book", "remotecontrol", "potato", "carrot", "bread", "milk", "wineglass",
        "cellphone", "toy", "spoon", "mug", "juice", "beer", "wine", "folder", "magazine", 
        "coffeepot", "glass", "pot", "notes", "check", "address_book"]

CHARACTER_LIST = [
    "character"
]

ALL_LIST = ROOM_LIST + SURFACE_LIST + CONTAINER_LIST + OBJECT_LIST + CHARACTER_LIST

ROOM_COMPONENTS = {
    "kitchen": ["fridge", "kitchentable", "kitchencabinet", "stove", "microwave"],
    "livingroom": ["sofa", "coffeetable", "desk"],
    "bathroom": ["bathroomcabinet", "bathroomcounter"],
    "bedroom": ["bed"]
}

ROOM_POSSIBILITY = {
    "kitchen": ["fridge", "kitchentable", "kitchencabinet", "stove", "microwave", "dishwasher", "kitchencounter"],
    "livingroom": ["sofa", "coffeetable", "desk", "cabinet"],
    "bathroom": ["bathroomcabinet", "bathroomcounter"],
    "bedroom": ["bed", "sofa", "coffeetable", "desk", "cabinet"]
}