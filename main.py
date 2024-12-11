import numpy as np
from models.floatbase import floatbase
from spatial.mcI import mcI

# Assuming mcI is defined elsewhere in your code
# def mcI(mass, dimensions, inertia):
#     # Placeholder for the inertia calculation function
#     # This function should return the inertia matrix for the given parameters
#     return inertia  # Replace with actual inertia calculation logic


def singlebody():
    """
    Create a model data structure for a single free-floating,
    unit-mass rigid body having the shape of a rectangular box
    with dimensions of 3, 2, and 1 in the x, y, and z directions, respectively.

    Returns:
        model: A dictionary representing the single body model.
    """

    print("singlebody")

    # # Persistent storage for the model
    # if not hasattr(singlebody, "memory"):
    #     singlebody.memory = None  # Initialize memory

    # if singlebody.memory is not None:
    #     return singlebody.memory  # Return stored model if it exists

    # Create the model
    model = {}
    model['NB'] = 1
    model['parent'] = [0]
    model['jtype'] = ['R']  # Sacrificial joint replaced by floatbase
    model['Xtree'] = [np.eye(6)]
    # model['Xtree'] = np.eye(6)

    model['gravity'] = [0, 0, 0]  # Explicitly state zero gravity

    # Inertia of a uniform-density box with one vertex at (0,0,0)
    # and a diametrically opposite one at (3,2,1).
    model['I'] = [mcI(1, np.array([3, 2, 1]) / 2,
                      np.diag([4 + 1, 9 + 1, 9 + 4]) / 12)]

    # Appearance settings
    model['appearance'] = {}
    model['appearance']['base'] = [
        {'colour': [0.9, 0, 0], 'line': [[0, 0, 0], [2, 0, 0]]},
        {'colour': [0, 0.9, 0], 'line': [[0, 0, 0], [0, 2, 0]]},
        {'colour': [0, 0.3, 0.9], 'line': [[0, 0, 0], [0, 0, 2]]}
    ]

    # Draw the floating body
    model['appearance']['body'] = [{'box': [[0, 0, 0], [3, 2, 1]]}]

    # Replace joint 1 with a chain of 6 joints emulating a floating base
    model = floatbase(model)

    # Store the model in memory for future calls
    # singlebody.memory = model

    return model


def main():
    sb = singlebody()
    print(sb.keys())
    print(sb['NB'])


if __name__ == "__main__":
    main()
