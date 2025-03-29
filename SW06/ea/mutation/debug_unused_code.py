# import objgraph
# from numpy.random import choice
# from pympler import muppy, summary


# def debugMemory(round_nbr, log):
    # print("--- DEBUG --- : Memory consumption: {}".format(memory_usage_psutil()))
    # log.addToLog("--- DEBUG --- : Memory consumption after round {}: {}".format(round_nbr, memory_usage_psutil()))
    # all_objects = muppy.get_objects()
    # sum1 = summary.summarize(all_objects)
    # summary.print_(sum1)
    # objgraph.show_most_common_types()
    # objgraph.show_backrefs(choice(objgraph.by_type('tuple')), filename="tuple_refs_{}.png".format(round_nbr))
