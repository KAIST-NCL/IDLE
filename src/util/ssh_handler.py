import paramiko
import re
import os

class shell_handler:

    def __init__(self, host):
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(host)

        channel = self.ssh.invoke_shell()
        self.stdin = channel.makefile('wb')
        self.stdout = channel.makefile('r')

    def __del__(self):
        self.ssh.close()

    def execute(self, cmd):
        """

        :param cmd: the command to be executed on the remote computer
        :examples:  execute('ls')
                    execute('finger')
                    execute('cd folder_name')
        """
        cmd = cmd.strip('\n')
        self.stdin.write(cmd + '\n')
        finish = 'end of stdOUT buffer. finished with exit status'
        echo_cmd = 'echo {} $?'.format(finish)
        self.stdin.write(echo_cmd + '\n')
        shin = self.stdin
        self.stdin.flush()

        shout = []
        sherr = []
        exit_status = 0
        for line in self.stdout:
            if str(line).startswith(cmd) or str(line).startswith(echo_cmd):
                # up for now filled with shell junk from stdin
                shout = []
            elif str(line).startswith(finish):
                # our finish command ends with the exit status
                exit_status = int(str(line).rsplit(maxsplit=1)[1])
                if exit_status:
                    # stderr is combined with stdout.
                    # thus, swap sherr with shout in a case of failure.
                    sherr = shout
                    shout = []
                break
            else:
                # get rid of 'coloring and formatting' special characters
                shout.append(re.compile(r'(\x9B|\x1B\[)[0-?]*[ -/]*[@-~]').sub('', line).
                             replace('\b', '').replace('\r', ''))
                # print shout[-1]

        # first and last lines of shout/sherr contain a prompt
        if shout and echo_cmd in shout[-1]:
            shout.pop()
        if shout and cmd in shout[0]:
            shout.pop(0)
        if sherr and echo_cmd in sherr[-1]:
            sherr.pop()
        if sherr and cmd in sherr[0]:
            sherr.pop(0)

        return shin, shout, sherr

class sftp_handler:

    def __init__(self, host):
        paramiko.util.log_to_file('/tmp/paramiko.log')
        self.t = paramiko.Transport(host)
        self.t.connect(username='hduser', password='1234555')
        self.sftp = paramiko.SFTPClient.from_transport(self.t)

    def __del__(self):
        self.sftp.close()
        self.t.close()

    def upload(self, local_path, remote_path):
        print 'Upload %s -> %s' % (local_path, remote_path)
        self.sftp.put(local_path, remote_path)

    def uploadRecursive(self, local_path, remote_path):
        ''' Uploads the contents of the source directory to the target path. The
            target directory needs to exists. All subdirectories in source are
            created under target.
        '''
        self.mkdir(remote_path, ignore_existing=True)
        for item in os.listdir(local_path):
            if os.path.isfile(os.path.join(local_path, item)):
                self.upload(os.path.join(local_path, item), '%s/%s' % (remote_path, item))
            else:
                self.mkdir('%s/%s' % (remote_path, item), ignore_existing=True)
                self.uploadRecursive(os.path.join(local_path, item), '%s/%s' % (remote_path, item))

    def mkdir(self, path, mode=511, ignore_existing=False):
        ''' Augments mkdir by adding an option to not fail if the folder exists  '''
        try:
            self.sftp.mkdir(path, mode)
        except IOError:
            if ignore_existing:
                pass
            else:
                raise