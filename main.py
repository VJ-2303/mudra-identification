import cv2
import mediapipe as mp
import math

# ----------------------------
# Helpers: distance / normalization / precompute
# ----------------------------
def get_distance(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

def get_scale_ref(landmarks):
    # palm width: wrist (0) to middle_mcp (9)
    return get_distance(landmarks[0], landmarks[9]) + 1e-6

def compute_distance_tables(landmarks):
    """
    Compute raw distances and normalized distances between all landmark pairs.
    Returns (dist_table, ndist_table, scale_ref)
    dist_table keys: (i, j) with i < j
    """
    scale_ref = get_scale_ref(landmarks)
    dist = {}
    ndist = {}
    for i in range(21):
        for j in range(i + 1, 21):
            d = math.sqrt((landmarks[i].x - landmarks[j].x)**2 +
                          (landmarks[i].y - landmarks[j].y)**2)
            dist[(i, j)] = d
            # normalized by palm width
            ndist[(i, j)] = d / scale_ref
    return dist, ndist, scale_ref

def _key(i, j):
    return (i, j) if i < j else (j, i)

def raw_dist(i, j, dist_table):
    return dist_table.get(_key(i, j), 0.0)

def norm_dist(i, j, ndist_table):
    return ndist_table.get(_key(i, j), 0.0)

# Angle helper (works with landmark coords, scale-invariant)
def angle_between(v1, v2):
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    if mag1 == 0 or mag2 == 0:
        return None
    cos_a = (v1[0]*v2[0] + v1[1]*v2[1]) / (mag1 * mag2)
    # clamp numerically
    cos_a = max(-1.0, min(1.0, cos_a))
    return math.degrees(math.acos(cos_a))

# ----------------------------
# Straightness check (normalized)
# ----------------------------
def is_finger_straight(landmarks, dist_table, ndist_table, mcp_index, pip_index, tip_index, threshold=0.9):
    """
    Normalized straightness: uses normalized distances (divided by palm width).
    straightness_ratio = (MCP->TIP) / (MCP->PIP + PIP->TIP)
    Return True if ratio > threshold (threshold tuned as per your original values).
    """
    mcp_pip = norm_dist(mcp_index, pip_index, ndist_table)
    pip_tip = norm_dist(pip_index, tip_index, ndist_table)
    mcp_tip = norm_dist(mcp_index, tip_index, ndist_table)

    total = mcp_pip + pip_tip
    if total == 0:
        return False
    straightness_ratio = mcp_tip / total
    return straightness_ratio > threshold

# ----------------------------
# Mudra functions (updated signatures use dist_table + ndist_table)
# ----------------------------
def is_pataka_mudra(landmarks, dist_table, ndist_table, scale_ref):
    # All four fingers straight + thumb tucked (normalized)
    index_straight = is_finger_straight(landmarks, dist_table, ndist_table, 5, 6, 8, threshold=0.97)
    middle_straight = is_finger_straight(landmarks, dist_table, ndist_table, 9, 10, 12, threshold=0.97)
    ring_straight = is_finger_straight(landmarks, dist_table, ndist_table, 13, 14, 16, threshold=0.97)
    pinky_straight = is_finger_straight(landmarks, dist_table, ndist_table, 17, 18, 20, threshold=0.97)
    if not (index_straight and middle_straight and ring_straight and pinky_straight):
        return False
    thumb_index_nd = norm_dist(4, 5, ndist_table)
    mcp_nd = norm_dist(5, 9, ndist_table)
    thumb_tucked = thumb_index_nd < (mcp_nd * 1.5)
    return thumb_tucked

def is_tripataka_mudra(landmarks, dist_table, ndist_table, scale_ref):
    index_straight = is_finger_straight(landmarks, dist_table, ndist_table, 5, 6, 8)
    middle_straight = is_finger_straight(landmarks, dist_table, ndist_table, 9, 10, 12)
    pinky_straight = is_finger_straight(landmarks, dist_table, ndist_table, 17, 18, 20)
    if not (index_straight and middle_straight and pinky_straight):
        return False
    ring_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 13, 14, 16, threshold=0.8)
    if not ring_bent:
        return False
    thumb_index_nd = norm_dist(4, 5, ndist_table)
    mcp_nd = norm_dist(5, 9, ndist_table)
    thumb_tucked = thumb_index_nd < (mcp_nd * 1.5)
    return thumb_tucked

def is_mayura_mudra(landmarks, dist_table, ndist_table, scale_ref):
    index_straight = is_finger_straight(landmarks, dist_table, ndist_table, 5, 6, 8)
    middle_straight = is_finger_straight(landmarks, dist_table, ndist_table, 9, 10, 12)
    pinky_straight = is_finger_straight(landmarks, dist_table, ndist_table, 17, 18, 20)
    if not (index_straight and middle_straight and pinky_straight):
        return False
    ring_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 13, 14, 16, threshold=0.8)
    if not ring_bent:
        return False
    thumb_ring_nd = norm_dist(4, 16, ndist_table)
    mcp_nd = norm_dist(5, 9, ndist_table)
    tips_touching = thumb_ring_nd < (mcp_nd * 1.0)
    return tips_touching

def is_ardha_chandra_mudra(landmarks, dist_table, ndist_table, scale_ref):
    # All five fingers straight (thumb slightly lower threshold)
    thumb_straight = is_finger_straight(landmarks, dist_table, ndist_table, 2, 3, 4, threshold=0.85)
    index_straight = is_finger_straight(landmarks, dist_table, ndist_table, 5, 6, 8)
    middle_straight = is_finger_straight(landmarks, dist_table, ndist_table, 9, 10, 12)
    ring_straight = is_finger_straight(landmarks, dist_table, ndist_table, 13, 14, 16)
    pinky_straight = is_finger_straight(landmarks, dist_table, ndist_table, 17, 18, 20)
    if not (thumb_straight and index_straight and middle_straight and ring_straight and pinky_straight):
        return False

    # Check perpendicular (L-shape) using vectors (angle near 90 deg)
    thumb_mcp = landmarks[2]
    thumb_tip = landmarks[4]
    index_mcp = landmarks[5]
    index_tip = landmarks[8]
    v_thumb = (thumb_tip.x - thumb_mcp.x, thumb_tip.y - thumb_mcp.y)
    v_index = (index_tip.x - index_mcp.x, index_tip.y - index_mcp.y)
    ang = angle_between(v_thumb, v_index)
    if ang is None:
        return False
    # allow wide tolerance: want near 90 degrees (60..120)
    if not (60 <= ang <= 120):
        return False

    # Fingers together check (normalized distances between tips)
    index_middle_nd = norm_dist(8, 12, ndist_table)
    middle_ring_nd = norm_dist(12, 16, ndist_table)
    ring_pinky_nd = norm_dist(16, 20, ndist_table)
    mcp_nd = norm_dist(5, 9, ndist_table)
    threshold = mcp_nd * 1.5
    return (index_middle_nd < threshold and middle_ring_nd < threshold and ring_pinky_nd < threshold)

def is_trishula_mudra(landmarks, dist_table, ndist_table, scale_ref):
    index_straight = is_finger_straight(landmarks, dist_table, ndist_table, 5, 6, 8)
    middle_straight = is_finger_straight(landmarks, dist_table, ndist_table, 9, 10, 12)
    ring_straight = is_finger_straight(landmarks, dist_table, ndist_table, 13, 14, 16)
    if not (index_straight and middle_straight and ring_straight):
        return False
    pinky_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 17, 18, 20, threshold=0.8)
    if not pinky_bent:
        return False
    thumb_pinky_nd = norm_dist(4, 20, ndist_table)
    mcp_nd = norm_dist(5, 9, ndist_table)
    return thumb_pinky_nd < (mcp_nd * 1.0)

def is_arala_mudra(landmarks, dist_table, ndist_table, scale_ref):
    middle_straight = is_finger_straight(landmarks, dist_table, ndist_table, 9, 10, 12, threshold=0.88)
    ring_straight = is_finger_straight(landmarks, dist_table, ndist_table, 13, 14, 16, threshold=0.88)
    pinky_straight = is_finger_straight(landmarks, dist_table, ndist_table, 17, 18, 20, threshold=0.88)
    index_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 5, 6, 8, threshold=0.92)
    if not (middle_straight and ring_straight and pinky_straight and index_bent):
        return False
    thumb_index_nd = norm_dist(4, 8, ndist_table)
    mcp_nd = norm_dist(5, 9, ndist_table)
    thumb_touch = thumb_index_nd < (mcp_nd * 1.2)
    # orientation check (thumb across palm)
    thumb_ip = landmarks[3]
    index_pip = landmarks[6]
    thumb_to_index_angle = abs(thumb_ip.x - index_pip.x)
    thumb_correct_angle = thumb_to_index_angle > 0.05
    return thumb_touch and thumb_correct_angle

def is_kartari_mukham_mudra(landmarks, dist_table, ndist_table, scale_ref):
    index_straight = is_finger_straight(landmarks, dist_table, ndist_table, 5, 6, 8)
    middle_straight = is_finger_straight(landmarks, dist_table, ndist_table, 9, 10, 12)
    if not (index_straight and middle_straight):
        return False
    # ensure extended away from wrist
    index_tip_wrist_nd = norm_dist(8, 0, ndist_table)
    index_mcp_wrist_nd = norm_dist(5, 0, ndist_table)
    middle_tip_wrist_nd = norm_dist(12, 0, ndist_table)
    middle_mcp_wrist_nd = norm_dist(9, 0, ndist_table)
    if not (index_tip_wrist_nd > index_mcp_wrist_nd and middle_tip_wrist_nd > middle_mcp_wrist_nd):
        return False
    ring_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 13, 14, 16, threshold=0.8)
    pinky_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 17, 18, 20, threshold=0.8)
    if not (ring_bent and pinky_bent):
        return False
    thumb_ring_pip_nd = norm_dist(4, 14, ndist_table)
    thumb_ring_tip_nd = norm_dist(4, 16, ndist_table)
    mcp_nd = norm_dist(5, 9, ndist_table)
    touch_threshold = mcp_nd * 2.0
    return (thumb_ring_pip_nd < touch_threshold) or (thumb_ring_tip_nd < touch_threshold)

def is_shuka_tundam_mudra(landmarks, dist_table, ndist_table, scale_ref):
    middle_straight = is_finger_straight(landmarks, dist_table, ndist_table, 9, 10, 12)
    pinky_straight = is_finger_straight(landmarks, dist_table, ndist_table, 17, 18, 20)
    index_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 5, 6, 8, threshold=0.7)
    ring_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 13, 14, 16, threshold=0.7)
    if not (middle_straight and pinky_straight and index_bent and ring_bent):
        return False
    thumb_index_nd = norm_dist(4, 5, ndist_table)
    mcp_nd = norm_dist(5, 9, ndist_table)
    return thumb_index_nd < (mcp_nd * 1.5)

def is_musthi_mudra(landmarks, dist_table, ndist_table, scale_ref):
    index_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 5, 6, 8, threshold=0.8)
    middle_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 9, 10, 12, threshold=0.8)
    ring_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 13, 14, 16, threshold=0.8)
    pinky_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 17, 18, 20, threshold=0.8)
    if not (index_bent and middle_bent and ring_bent and pinky_bent):
        return False
    thumb_index_pip_nd = norm_dist(4, 6, ndist_table)
    thumb_middle_pip_nd = norm_dist(4, 10, ndist_table)
    mcp_nd = norm_dist(5, 9, ndist_table)
    touch_threshold = mcp_nd * 1.5
    return (thumb_index_pip_nd < touch_threshold) or (thumb_middle_pip_nd < touch_threshold)

def is_shikharam_mudra(landmarks, dist_table, ndist_table, scale_ref):
    index_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 5, 6, 8, threshold=0.8)
    middle_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 9, 10, 12, threshold=0.8)
    ring_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 13, 14, 16, threshold=0.8)
    pinky_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 17, 18, 20, threshold=0.8)
    if not (index_bent and middle_bent and ring_bent and pinky_bent):
        return False
    thumb_straight = is_finger_straight(landmarks, dist_table, ndist_table, 2, 3, 4, threshold=0.85)
    if not thumb_straight:
        return False
    thumb_index_nd = norm_dist(4, 6, ndist_table)
    thumb_middle_nd = norm_dist(4, 10, ndist_table)
    mcp_nd = norm_dist(5, 9, ndist_table)
    touch_threshold = mcp_nd * 1.5
    thumb_is_tucked = (thumb_index_nd < touch_threshold or thumb_middle_nd < touch_threshold)
    return not thumb_is_tucked

def is_chandrakala_mudra(landmarks, dist_table, ndist_table, scale_ref):
    index_straight = is_finger_straight(landmarks, dist_table, ndist_table, 5, 6, 8)
    thumb_straight = is_finger_straight(landmarks, dist_table, ndist_table, 2, 3, 4, threshold=0.85)
    middle_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 9, 10, 12, threshold=0.8)
    ring_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 13, 14, 16, threshold=0.8)
    pinky_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 17, 18, 20, threshold=0.8)
    if not (index_straight and thumb_straight and middle_bent and ring_bent and pinky_bent):
        return False
    v_thumb = (landmarks[4].x - landmarks[2].x, landmarks[4].y - landmarks[2].y)
    v_index = (landmarks[8].x - landmarks[5].x, landmarks[8].y - landmarks[5].y)
    ang = angle_between(v_thumb, v_index)
    if ang is None:
        return False
    # wide spread: angle > ~45
    return ang > 45

def is_suchi_mudra(landmarks, dist_table, ndist_table, scale_ref):
    index_straight = is_finger_straight(landmarks, dist_table, ndist_table, 5, 6, 8, threshold=0.95)
    # ensure extended away from wrist
    index_tip_wrist_nd = norm_dist(8, 0, ndist_table)
    index_mcp_wrist_nd = norm_dist(5, 0, ndist_table)
    index_extended = index_tip_wrist_nd > index_mcp_wrist_nd
    middle_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 9, 10, 12, threshold=0.8)
    ring_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 13, 14, 16, threshold=0.8)
    pinky_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 17, 18, 20, threshold=0.8)
    if not (index_straight and index_extended and middle_bent and ring_bent and pinky_bent):
        return False
    thumb_middle_nd = norm_dist(4, 10, ndist_table)
    mcp_nd = norm_dist(5, 9, ndist_table)
    touch_threshold = mcp_nd * 0.9
    return thumb_middle_nd < touch_threshold

def is_ardhapataka_mudra(landmarks, dist_table, ndist_table, scale_ref):
    index_straight = is_finger_straight(landmarks, dist_table, ndist_table, 5, 6, 8)
    middle_straight = is_finger_straight(landmarks, dist_table, ndist_table, 9, 10, 12)
    if not (index_straight and middle_straight):
        return False
    ring_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 13, 14, 16, threshold=0.8)
    pinky_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 17, 18, 20, threshold=0.8)
    if not (ring_bent and pinky_bent):
        return False
    thumb_index_nd = norm_dist(4, 5, ndist_table)
    mcp_nd = norm_dist(5, 9, ndist_table)
    return thumb_index_nd < (mcp_nd * 1.5)

def is_kapitha_mudra(landmarks, dist_table, ndist_table, scale_ref):
    index_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 5, 6, 8, threshold=0.92)
    middle_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 9, 10, 12, threshold=0.92)
    ring_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 13, 14, 16, threshold=0.92)
    pinky_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 17, 18, 20, threshold=0.92)
    if not (index_bent and middle_bent and ring_bent and pinky_bent):
        return False
    thumb_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 2, 3, 4, threshold=0.92)
    if not thumb_bent:
        return False
    thumb_index_nd = norm_dist(4, 8, ndist_table)
    thumb_middle_nd = norm_dist(4, 12, ndist_table)
    mcp_nd = norm_dist(5, 9, ndist_table)
    touch_threshold = mcp_nd * 1.5
    return thumb_index_nd < touch_threshold or thumb_middle_nd < touch_threshold
def is_katakamukha_mudra(landmarks, dist_table, ndist_table, scale_ref):
    """
    Katakamukha Mudra (refined to prevent conflict with Hamsapaksha)
    ✔ Ring and Pinky straight
    ✔ Thumb tip touches BOTH Index & Middle tips (mandatory)
    """
    # Ring, Pinky straight
    ring_straight  = is_finger_straight(landmarks, dist_table, ndist_table, 13, 14, 16, threshold=0.83)
    pinky_straight = is_finger_straight(landmarks, dist_table, ndist_table, 17, 18, 20, threshold=0.83)
    if not (ring_straight and pinky_straight):
        return False

    # Thumb tip must touch BOTH Index tip AND Middle tip
    thumb_index_nd  = norm_dist(4, 8, ndist_table)
    thumb_middle_nd = norm_dist(4, 12, ndist_table)
    mcp_nd = norm_dist(5, 9, ndist_table)

    touch_threshold = mcp_nd * 0.55  # tightened to avoid false positives

    if not (thumb_index_nd < touch_threshold and thumb_middle_nd < touch_threshold):
        return False

    return True


def is_padmakosha_mudra(landmarks, dist_table, ndist_table, scale_ref):
    index_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 5, 6, 8, threshold=0.90)
    middle_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 9, 10, 12, threshold=0.90)
    ring_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 13, 14, 16, threshold=0.90)
    pinky_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 17, 18, 20, threshold=0.90)
    if not (index_bent and middle_bent and ring_bent and pinky_bent):
        return False
    dist_index_middle = norm_dist(8, 12, ndist_table)
    dist_middle_ring = norm_dist(12, 16, ndist_table)
    dist_ring_pinky = norm_dist(16, 20, ndist_table)
    mcp_nd = norm_dist(5, 9, ndist_table)
    tip_closeness_threshold = mcp_nd * 1.0
    tips_close = (dist_index_middle < tip_closeness_threshold and
                  dist_middle_ring < tip_closeness_threshold and
                  dist_ring_pinky < tip_closeness_threshold)
    if not tips_close:
        return False
    thumb_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 2, 3, 4, threshold=0.90)
    return thumb_bent

def is_sarpashirsha_mudra(landmarks, dist_table, ndist_table, scale_ref):
    loose = 0.80
    idx = is_finger_straight(landmarks, dist_table, ndist_table, 5, 6, 8, threshold=loose)
    mid = is_finger_straight(landmarks, dist_table, ndist_table, 9, 10, 12, threshold=loose)
    rng = is_finger_straight(landmarks, dist_table, ndist_table, 13, 14, 16, threshold=loose)
    pnk = is_finger_straight(landmarks, dist_table, ndist_table, 17, 18, 20, threshold=loose)
    if not (idx and mid and rng and pnk):
        return False
    tip_dist = norm_dist(8, 20, ndist_table)
    mcp_dist = norm_dist(5, 17, ndist_table)
    if not (tip_dist < mcp_dist):
        return False
    thumb_index_nd = norm_dist(4, 5, ndist_table)
    mcp_ref_nd = norm_dist(5, 9, ndist_table)
    return thumb_index_nd < (mcp_ref_nd * 1.5)

def is_mrigasheersha_mudra(landmarks, dist_table, ndist_table, scale_ref):
    pinky_straight = is_finger_straight(landmarks, dist_table, ndist_table, 17, 18, 20, threshold=0.85)
    thumb_straight = is_finger_straight(landmarks, dist_table, ndist_table, 2, 3, 4, threshold=0.80)
    index_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 5, 6, 8, threshold=0.85)
    middle_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 9, 10, 12, threshold=0.85)
    ring_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 13, 14, 16, threshold=0.85)
    if not (pinky_straight and thumb_straight and index_bent and middle_bent and ring_bent):
        return False
    v_thumb = (landmarks[4].x - landmarks[2].x, landmarks[4].y - landmarks[2].y)
    v_hand = (landmarks[9].x - landmarks[0].x, landmarks[9].y - landmarks[0].y)
    ang = angle_between(v_thumb, v_hand)
    if ang is None:
        return False
    return ang > 45  # wide thumb separation

def is_simhamukha_mudra(landmarks, dist, ndist, scale):
    # Straight index & pinky
    index_straight = is_finger_straight(landmarks, dist, ndist, 5,6,8, 0.82)
    pinky_straight = is_finger_straight(landmarks, dist, ndist, 17,18,20, 0.80)

    if not (index_straight and pinky_straight):
        return False

    # Middle & Ring half-bent
    middle_half = (not is_finger_straight(landmarks, dist, ndist, 9,10,12, 0.92))
    ring_half   = (not is_finger_straight(landmarks, dist, ndist, 13,14,16, 0.92))

    if not (middle_half and ring_half):
        return False

    # Pinky must be vertical (upwards) to separate from Hamsapaksha
    pinky_tip = landmarks[20]
    pinky_mcp = landmarks[17]
    if not (pinky_tip.y < pinky_mcp.y - scale*0.20):
        return False

    # Thumb inside
    thumb_mid = norm_dist(4, 10, ndist)
    return thumb_mid < (scale * 3.5)




def is_chatura_mudra(landmarks, distances, ndistances, scale_ref):
    """
    Strict + Slightly Flexible Chatura Mudra Detection
    """

    # 1) All four fingers must be straight
    if not (
        is_finger_straight(landmarks, distances, ndistances, 5, 6, 8, 0.85) and
        is_finger_straight(landmarks, distances, ndistances, 9, 10, 12, 0.85) and
        is_finger_straight(landmarks, distances, ndistances, 13, 14, 16, 0.85) and
        is_finger_straight(landmarks, distances, ndistances, 17, 18, 20, 0.85)
    ):
        return False

    # 2) Index-Middle-Ring must be close
    index_tip  = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip   = landmarks[16]

    d_im = math.dist((index_tip.x, index_tip.y), (middle_tip.x, middle_tip.y))
    d_mr = math.dist((middle_tip.x, middle_tip.y), (ring_tip.x, ring_tip.y))

    if d_im > scale_ref * 0.30 or d_mr > scale_ref * 0.30:
        return False

    # Palm reference
    wrist = landmarks[0]
    middle_mcp = (landmarks[9].x, landmarks[9].y)
    palm_center = ((wrist.x + middle_mcp[0]) / 2, (wrist.y + middle_mcp[1]) / 2)

    # 3) Thumb inside palm
    thumb_tip = (landmarks[4].x, landmarks[4].y)
    index_mcp = (landmarks[5].x, landmarks[5].y)
    ring_mcp  = (landmarks[13].x, landmarks[13].y)
    pinky_mcp = (landmarks[17].x, landmarks[17].y)

    d_thumb_palm = math.dist(thumb_tip, palm_center)
    d_thumb_index = math.dist(thumb_tip, (landmarks[8].x, landmarks[8].y))
    d_thumb_middle = math.dist(thumb_tip, (landmarks[12].x, landmarks[12].y))
    d_thumb_ring = math.dist(thumb_tip, (landmarks[16].x, landmarks[16].y))
    d_thumb_pinky = math.dist(thumb_tip, (landmarks[20].x, landmarks[20].y))

    tolerance = scale_ref * 0.01  # allows little higher tuck
    thumb_depth_ok = thumb_tip[1] > (middle_mcp[1] - tolerance)
    thumb_x_inside = min(index_mcp[0], pinky_mcp[0]) < thumb_tip[0] < max(index_mcp[0], pinky_mcp[0])
    thumb_deep_inside = (
        d_thumb_palm < d_thumb_index and
        d_thumb_palm < d_thumb_middle and
        d_thumb_palm < d_thumb_ring and
        d_thumb_palm < d_thumb_pinky
    )

    if not (thumb_depth_ok and thumb_deep_inside and thumb_x_inside):
        return False

    return True



def is_bhramara_mudra(landmarks, distances, ndistances, scale_ref):
    """
    Bhramara Mudra (User-defined Logic)
    Conditions:
      1. Thumb tip touches Middle tip
      2. Ring and Pinky are straight
      3. Index is rolled: index tip touches index MCP
    """

    # -----------------------------
    # Rule 1: Thumb-tip & Middle-tip touching
    # -----------------------------
    thumb_tip = (landmarks[4].x, landmarks[4].y)
    middle_tip = (landmarks[12].x, landmarks[12].y)
    d_tm = math.dist(thumb_tip, middle_tip)

    if d_tm > scale_ref * 0.22:  # threshold for touching
        return False

    # -----------------------------
    # Rule 2: Ring & Pinky must be straight
    # -----------------------------
    ring_straight = is_finger_straight(landmarks, distances, ndistances, 13, 14, 16, 0.85)
    pinky_straight = is_finger_straight(landmarks, distances, ndistances, 17, 18, 20, 0.85)

    if not (ring_straight and pinky_straight):
        return False

    # -----------------------------
    # Rule 3: Index is rolled (tip touches its MCP)
    # -----------------------------
    index_tip = (landmarks[8].x, landmarks[8].y)
    index_mcp = (landmarks[5].x, landmarks[5].y)
    d_index_fold = math.dist(index_tip, index_mcp)

    if d_index_fold > scale_ref * 0.22:  # must touch MCP
        return False

    return True


def is_hamsasya_mudra(landmarks, dist_table, ndist_table, scale_ref):
    """
    Hamsasya Mudra

    1. Thumb tip close to Index tip (circle / OK sign).
    2. Index is bent (not straight).
    3. Middle, Ring, Pinky are straight (or mostly straight).
    """

    # ---- 1) Thumb tip & Index tip must be close ----
    d_thumb_index_tip = norm_dist(4, 8, ndist_table)   # thumb tip ↔ index tip
    d_thumb_index_pip = norm_dist(4, 6, ndist_table)   # thumb tip ↔ index PIP

    # tolerant threshold (because of rotation / perspective)
    if not (d_thumb_index_tip < 0.40 or d_thumb_index_pip < 0.40):
        return False

    # ---- 2) Index must be clearly bent (not straight) ----
    index_is_straight = is_finger_straight(
        landmarks, dist_table, ndist_table,
        5, 6, 8, threshold=0.88
    )
    if index_is_straight:
        # if index is straight, this is Pataka / other, not Hamsasya
        return False

    # ---- 3) Middle, Ring, Pinky should be straight / extended ----
    middle_straight = is_finger_straight(
        landmarks, dist_table, ndist_table,
        9, 10, 12, threshold=0.80
    )
    ring_straight = is_finger_straight(
        landmarks, dist_table, ndist_table,
        13, 14, 16, threshold=0.80
    )
    pinky_straight = is_finger_straight(
        landmarks, dist_table, ndist_table,
        17, 18, 20, threshold=0.78   # slightly more relaxed
    )

    if not (middle_straight and ring_straight and pinky_straight):
        return False

    return True

def is_hamsapaksha_mudra(landmarks, dist, ndist, scale):
    
    # Y coordinates (smaller y = higher on screen)
    y_i = landmarks[8].y
    y_m = landmarks[12].y
    y_r = landmarks[16].y
    y_p = landmarks[20].y

    # ---------- 1) Pinky highest check (slightly relaxed) ----------
    min_gap = scale * 0.06   # was 0.08, now more tolerant
    if not (y_p + min_gap < min(y_r, y_m, y_i)):
        return False

    # ---------- 2) Index/Middle/Ring same height group (wider tolerance) ----------
    row_tol = scale * 0.16   # was 0.10
    if not (abs(y_i - y_m) < row_tol and
            abs(y_m - y_r) < row_tol and
            abs(y_i - y_r) < row_tol):
        return False

    # ---------- 3) Thumb must NOT touch index/middle (relaxed to avoid false rejecting) ----------
    thumb_i = norm_dist(4, 8, ndist)
    thumb_m = norm_dist(4, 12, ndist)

    # Was 0.28, now slightly relaxed so small variations won't break detection
    if thumb_i < scale * 0.25 or thumb_m < scale * 0.25:
        return False

    return True

def is_mukula_mudra(landmarks, dist, ndist, scale):
    import math

    tips = [4, 8, 12, 16, 20]
    pts = [(landmarks[t].x, landmarks[t].y) for t in tips]

    # Fingertip cluster center
    cx = sum([p[0] for p in pts]) / 5
    cy = sum([p[1] for p in pts]) / 5

    # Distances of tips from center
    dists = [math.dist(p, (cx, cy)) for p in pts]
    max_d = max(dists)

    # Allow wider cluster (improved tolerance)
    if max_d > scale * 2.2:   # was 1.5 before
        return False

    # Fingers must be bent (not straight)
    def bent(mcp, pip, tip):
        return not is_finger_straight(landmarks, dist, ndist, mcp, pip, tip, threshold=0.93)

    if not (
        bent(2, 3, 4) and
        bent(5, 6, 8) and
        bent(9, 10, 12) and
        bent(13, 14, 16) and
        bent(17, 18, 20)
    ):
        return False

    # Pairwise distances between tips (weakened rules)
    pairs = [(4, 8), (8, 12), (12, 16), (16, 20)]
    for (a, b) in pairs:
        if norm_dist(a, b, ndist) > scale * 1.25:  # was 0.75 before
            return False

    return True






# ----------------------------
# Mudra function registry (ordered by priority)
# ----------------------------
mudra_functions = {
    "Ardha Chandra Mudra": is_ardha_chandra_mudra,
    "Suchi Mudra": is_suchi_mudra,
    "Ardhapataka Mudra": is_ardhapataka_mudra,
    "Mayura Mudra": is_mayura_mudra,
    "Trishula Mudra": is_trishula_mudra,
    "Tripataka Mudra": is_tripataka_mudra,
    "Sarpashirsha Mudra": is_sarpashirsha_mudra,
    "Pataka Mudra": is_pataka_mudra,
    "Arala Mudra": is_arala_mudra,
    "Kartari Mukham Mudra": is_kartari_mukham_mudra,
    "Shuka Tundam Mudra": is_shuka_tundam_mudra,
    "Shikharam Mudra": is_shikharam_mudra,
    "Musthi Mudra": is_musthi_mudra,
    "Chandrakala Mudra": is_chandrakala_mudra,
    "Kapitha Mudra": is_kapitha_mudra,
    "Katakamukha Mudra": is_katakamukha_mudra,
    "Mrigasheersha Mudra": is_mrigasheersha_mudra,
    "Simhamukha Mudra": is_simhamukha_mudra,
    "Padmakosha Mudra": is_padmakosha_mudra,
    "Suchi (Needle) Mudra": is_suchi_mudra,  # duplicate alias if needed
    "Chatura Mudra": is_chatura_mudra,
    "Bhramara Mudra": is_bhramara_mudra,
    "Hamsasya Mudra": is_hamsasya_mudra,
    "Hamsapaksha Mudra": is_hamsapaksha_mudra,
    "Mukula Mudra": is_mukula_mudra


}

# ----------------------------
# Main webcam loop with Mediapipe
# ----------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = hands.process(image_rgb)
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        mudra_status = "No Mudra Detected"
        text_color = (0, 0, 255)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark
            dist_table, ndist_table, scale_ref = compute_distance_tables(landmarks)

            # iterate mudra functions in order
            for name, func in mudra_functions.items():
                try:
                    if func(landmarks, dist_table, ndist_table, scale_ref):
                        mudra_status = f"{name} Detected"
                        # example color mapping (simple)
                        text_color = (0, 255, 0)
                        break
                except Exception as e:
                    # keep loop robust in case a function errors for a frame
                    # (shouldn't normally happen)
                    # print("Mudra func error:", name, e)
                    continue

        cv2.putText(
            image,
            mudra_status,
            (50, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            text_color,
            3,
            cv2.LINE_AA)

        cv2.imshow('Mudra Detector', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
