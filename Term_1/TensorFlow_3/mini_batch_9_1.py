import pprint


def batches(batch_size, features, labels):
    """
    Create batches of features and labels

    Parameters
    ----------
    batch_size: The batch size
    features: List of features
    labels: List of labels

    Returns
    -------
        Batches of (Features, Labels) : ndarray
    """
    assert len(features) == len(labels)
    outout_batches = []

    sample_size = len(features)
    for start_i in range(0, sample_size, batch_size):
        end_i = start_i + batch_size
        batch = [features[start_i:end_i], labels[start_i:end_i]]
        outout_batches.append(batch)

    return outout_batches


def run():
    """
    Pretty prints the list into batches, for a double list.
    """

    # 4 Samples of features
    example_features = [
        ['F11', 'F12', 'F13', 'F14'],
        ['F21', 'F22', 'F23', 'F24'],
        ['F31', 'F32', 'F33', 'F34'],
        ['F41', 'F42', 'F43', 'F44']]
    # 4 Samples of labels
    example_labels = [
        ['L11', 'L12'],
        ['L21', 'L22'],
        ['L31', 'L32'],
        ['L41', 'L42']]
    pp = pprint.PrettyPrinter(depth=10, width=50)
    pp.pprint(batches(3, example_features, example_labels))

if __name__ == '__main__':
    run()
