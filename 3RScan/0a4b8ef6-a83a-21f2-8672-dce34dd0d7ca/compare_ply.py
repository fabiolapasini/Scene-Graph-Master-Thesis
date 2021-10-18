import point_cloud_utils as pcu

v1, f1, n1, uv1 = pcu.read_ply("labels.instances.align.annotated.v2ply")
v2, f2, n2, uv2 = pcu.read_ply("labels.instances.annotated.v2.ply")
d = pcu.pairwise_distances(v1, v2)