import numpy as np



def G2C_RGF(longitude, latitude, h) :
  GRS_a = 6378137
  GRS_f = 1/298.257222101
  GRS_b = GRS_a*(1-GRS_f)
  GRS_e = np.sqrt((GRS_a**2 - GRS_b**2) / GRS_a**2)
  N = GRS_a / np.sqrt(1 - GRS_e**2 * np.sin(latitude)**2)

  # coordinates in RGF93 frame
  X = (N + h) * np.cos(latitude) * np.cos(longitude)
  Y = (N + h) * np.cos(latitude) * np.sin(longitude)
  Z = ((1 - np.power(GRS_e,2)) * N + h) * np.sin(latitude)

  return np.array([X, Y, Z])

def C_RGF2ENU(P_ECEF, l, phi, h) :
  eO = G2C_RGF(l, phi, h)
  P_ECEF = P_ECEF[np.newaxis].T # (3,1)
  # rotation matrix from ECEF to ENU
  oAe = np.array([[-np.sin(l), np.cos(l), 0],
                    [-np.sin(phi)*np.cos(l), -np.sin(phi)*np.sin(l), np.cos(phi)],
                    [np.cos(phi)*np.cos(l), np.cos(phi)*np.sin(l), np.sin(phi)]])
  # ENU coordinates (East, North, Up), that is the local frame
  P_ENU = oAe @ (P_ECEF -  eO[np.newaxis].T)

  return P_ENU[:, 0]



def gdfs_to_local(nodes_proj, path):
  l = nodes_proj.at[nodes_proj.index[0],"x"] * np.pi / 180
  phi = nodes_proj.at[nodes_proj.index[0],"y"] * np.pi / 180
  h0 = 342.

  # put the points in a frame in meter :
  waypoints_ECEF = np.zeros((nodes_proj.loc[path]["x"].size, 3))
  waypoints_ENU = np.zeros((nodes_proj.loc[path]["x"].size, 3))
  
  for i in range(nodes_proj.loc[path]["x"].size):
      longitude = nodes_proj.loc[path[i]]["x"] * np.pi / 180
      latitude =  nodes_proj.loc[path[i]]["y"] * np.pi / 180

      # compute the coordinates in the RGF93 frame
      waypoints_ECEF[i] = G2C_RGF(longitude, latitude, h0)
      # compute the coordinates in the local frame
      waypoints_ENU[i] = C_RGF2ENU(waypoints_ECEF[i], l, phi, h0)

  return waypoints_ENU

