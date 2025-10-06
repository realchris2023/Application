import numpy as np

def normalize(x):
    """Normalizes a vector x to its unit vector (length 1)."""
    return x / np.linalg.norm(x)

def calculate_gains(c1, c2, v):
    """Calculate VBAP gains with phase inversion for beyond-speaker positions."""
    # Normalize vectors
    l1 = normalize(c1)  # Normalize the first speaker vector
    l2 = normalize(c2)  # Normalize the second speaker vector
    p = normalize(v)    # Normalize the virtual source vector 
    
    # Base matrix for VBAP
    L = np.column_stack([l1, l2])  # Create a matrix with the normalized speaker vectors as columns
    
    try:
        # Get standard VBAP gains
        g = np.linalg.solve(L, p)  # Solve the linear equation L * g = p for g
        # Normalize gains while preserving their signs.
        g = normalize(g)  # Calculate the magnitude of the gain vector g=[g1,g2],  sqrt((g1^2) + (g2^2))
        return g
        
    except np.linalg.LinAlgError:
        return np.array([0.707, 0.707])  # Return default gains if the matrix is singular - (sqrrt(2))/2


