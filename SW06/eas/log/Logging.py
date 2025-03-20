import datetime
from log.DNA_log import init_dna_log
from dna.DNA import DNA
import os
import shutil
import zipfile

class MainLog(object):
    # Init function
    def __init__(self, restart=False):
        if not restart:

            try:
                # Check for old log files an copy them away before start new log:
                # Find out date of old run:
                dateOldLogs = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
                wd = os.getcwd()
                dest = "log/OldLogs{}".format(dateOldLogs)
                os.mkdir(dest)

                srcs = "log/training_logs"
                if os.path.isdir(srcs)==False:
                    # make directory if not existing (after checkout)
                    os.mkdir(srcs)
                shutil.move(srcs, dest, shutil.copytree)
                os.mkdir(srcs)

                srcs = "log/training_models"
                if os.path.isdir(srcs)==False:
                    # make directory if not existing (after checkout)
                    os.mkdir(srcs)
                shutil.move(srcs, dest, shutil.copytree)
                os.mkdir(srcs)
                srcs = "".join([wd, "/log/main_log.txt"])
                shutil.copy(srcs, dest)
                srcs = "".join([wd, "/log/population_log.txt"])
                shutil.move(srcs, dest)
                srcs = "".join([wd, "/log/dna_log.txt"])
                shutil.move(srcs, dest)
                srcs = "".join([wd, "/log/arch_log.txt"])
                shutil.move(srcs, dest)
                srcs = "".join([wd, "/log/ranked_log.txt"])
                shutil.move(srcs, dest)
                srcs = "".join([wd, "/config.py"])
                shutil.copy(srcs, dest)
                srcs = "".join([wd, "/log/analysis.txt"])
                shutil.move(srcs, dest)

                # Get a list of all copied files
                paths = []
                for root, dirs, files in os.walk(dest):
                    for f_name in files:
                        path = os.path.join(root, f_name)  # get a file and add the total path
                        paths.append(path)

                # Zip them together
                zipf = zipfile.ZipFile(dest + ".zip", 'w', zipfile.ZIP_DEFLATED)
                # ziph is zipfile handle
                for file in paths:
                    zipf.write(file)
                zipf.close()

                #TODO: delete copied away file ?
            except FileNotFoundError:
                print("Could no copy logs")

            # Create a new log entry
            with open('./log/main_log.txt', 'a+') as myfile:
                myfile.write("\n")
                myfile.write("-------------------------------------------------------\n")
                myfile.write("New Log: ")
                myfile.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                myfile.write("\n")
            # Create a new log for the population log
            with open('./log/population_log.txt', 'a+') as myfile:
                myfile.write("\n")
                myfile.write("-------------------------------------------------------\n")
                myfile.write("New Population Log: ")
                myfile.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                myfile.write("\n")

            # Create a new log for the population log
            with open('./log/ranked_log.txt', 'a+') as myfile:
                myfile.write("\n")
                myfile.write("-------------------------------------------------------\n")
                myfile.write("New Population with Rank and Fitness Log: ")
                myfile.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                myfile.write("\n")
            # Also initialize DNA log
            init_dna_log()

    # Print the current date and time
    def printTimeToConsole(self):
        print("-------------------------------------------------------")
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("-------------------------------------------------------")

    def addToLog(self, text: str):
        print(text)
        with open('./log/main_log.txt', 'a') as myfile:
            myfile.write(text)
            myfile.write("\n")

    def addTimeToLog(self):
        self.printTimeToConsole()
        with open('./log/main_log.txt', 'a') as myfile:
            myfile.write("-------------------------------------------------------\n")
            myfile.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            myfile.write("-------------------------------------------------------\n")

    def addPopulationSet(self, round_nbr: int, pop_set: set):
        # Create list for DNA numbers
        dna_numbers = list()
        # Loop over DNAs in population and log the number in the list
        for dna in pop_set:
            assert isinstance(dna, DNA)
            dna_numbers.append(dna.dna_ID)
        # Create a string from the list
        dna_numbers_string = str(dna_numbers)
        # Write to log
        with open('./log/population_log.txt', 'a') as myfile:
            myfile.write("End of round {} - DNAs in Population: ".format(round_nbr))
            myfile.write(dna_numbers_string)
            myfile.write("\n")

    def addRankedPopulationWithFitness(self, round_nbr: int, pop_set: set):
        # Create list for DNA numbers
        dna_list = {}
        # Loop over DNAs in population and log the number in the list
        for dna in pop_set:
            assert isinstance(dna, DNA)
            dna_list[dna.rank] = dna
        # Create a string from the list
        dna_numbers_string = "["
        for rank in range(1, len(dna_list)+1):
            if rank in dna_list:
                dna_numbers_string += "Rank:{} ID:{} Fitness:{:.2f} \t".format(rank, dna_list[rank].dna_ID, dna_list[rank].fitness )
        dna_numbers_string += "]"
        # Write to log
        with open('./log/ranked_log.txt', 'a') as myfile:
            myfile.write("End of round {} - DNAs ranked in Population: ".format(round_nbr))
            myfile.write(dna_numbers_string)
            myfile.write("\n")