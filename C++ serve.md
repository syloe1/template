  **play**<br>
C++单独实现一次连接的tcp服务器， 使用socket, bind, listen, accept搞定。客户端不用写，先用telent直接连你的server测试。请给出C++代码实现。

	#include <iostream>
	#include <sys/socket.h>
	#include <netinet/in.h>
	#include <arpa/inet.h>
	#include <unistd.h>
	#include <string.h>
	int main() {
	  // 创建socket
	  int listenfd = socket(AF_INET, SOCK_STREAM, 0);
	  if (listenfd == -1) {
	    std::cerr << "Error creating socket" << std::endl;
	    return 1;
	  }
	
	  // 设置socket地址
	  struct sockaddr_in servaddr;
	  memset(&servaddr, 0, sizeof(servaddr));
	  servaddr.sin_family = AF_INET;
	  servaddr.sin_addr.s_addr = htonl(INADDR_ANY);
	  servaddr.sin_port = htons(8080);
	
	  // 绑定socket
	  if (bind(listenfd, (struct sockaddr*)&servaddr, sizeof(servaddr)) == -1) {
	    std::cerr << "Error binding socket" << std::endl;
	    return 1;
	  }

	  // 监听socket
	  if (listen(listenfd, 5) == -1) {
	    std::cerr << "Error listening on socket" << std::endl;
	    return 1;
	  }
	
	  std::cout << "Server is listening on port 8080" << std::endl;
	
	  // 接受连接
	  struct sockaddr_in clientaddr;
	  socklen_t clientaddr_len = sizeof(clientaddr);
	  int connfd = accept(listenfd, (struct sockaddr*)&clientaddr, &clientaddr_len);
	  if (connfd == -1) {
	    std::cerr << "Error accepting connection" << std::endl;
	    return 1;
	  }
	
	  std::cout << "Client connected" << std::endl;
	
	  // 读写数据
	  char buffer[1024];
	  while (true) {
	    memset(buffer, 0, sizeof(buffer));
	    int n = read(connfd, buffer, sizeof(buffer));
	    if (n == -1) {
	      std::cerr << "Error reading from socket" << std::endl;
	      break;
	    } else if (n == 0) {
	      std::cout << "Client disconnected" << std::endl;
	      break;
	    }
	    std::cout << "Received from client: " << buffer << std::endl;
	
	    // 回复数据
	    memset(buffer, 0, sizeof(buffer));
	    strcpy(buffer, "Hello, world!");
	    n = write(connfd, buffer, strlen(buffer));
	    if (n == -1) {
	      std::cerr << "Error writing to socket" << std::endl;
	      break;
	    }
	  }
	
	  // 关闭连接
	  close(connfd);
	  close(listenfd);
	
	  return 0;
	}
<br>

g++ a.cpp -o a
./a

telnet 192.168.71.136（ip) 8080 <br>
连接成功

netstat -anp | grep telent 查看pid<br>
kill -9 pid

	如果在一台服务器中测试， 需要关闭8080端口
<br>
C++编写简单的客户端，代替使用的telnet实现和服务器的通讯
	
	#include <iostream>
	#include <sys/types.h>
	#include <sys/socket.h>
	#include <netinet/in.h>
	#include <arpa/inet.h>
	#include <unistd.h>
	#include <cstring>
	
	using namespace std;
	
	int main() {
	  // 创建socket
	  int client_sockfd = socket(AF_INET, SOCK_STREAM, 0);
	  if (client_sockfd == -1) {
	    perror("socket");
	    return -1;
	  }
	
	  // 连接到服务器
	  struct sockaddr_in server_addr;
	  memset(&server_addr, 0, sizeof(server_addr));
	  server_addr.sin_family = AF_INET;
	  server_addr.sin_port = htons(8888); // 端口号
	  server_addr.sin_addr.s_addr = inet_addr("127.0.0.1"); // 服务器的IP地址
	
	  int connect_result = connect(client_sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr));
	  if (connect_result == -1) {
	    perror("connect");
	    return -1;
	  }
	
	  // 向服务器发送数据
	  char send_buffer[] = "Hello from client!";
	  int send_result = send(client_sockfd, send_buffer, strlen(send_buffer), 0);
	  if (send_result == -1) {
	    perror("send");
	    return -1;
	  }
	
	  // 接收服务器发送的数据
	  char recv_buffer[1024] = {0};
	  int recv_result = recv(client_sockfd, recv_buffer, 1024, 0);
	  if (recv_result == -1) {
	    perror("recv");
	    return -1;
	  }
	
	  // 输出服务器发送的数据
	  cout << "Received from server: " << recv_buffer << endl;
	
	  // 关闭socket
	  close(client_sockfd);
	
	  return 0;
	}
<br>
采用fork方式， 多进程条件下，实现多客户端的支持。请给出C++实现代码。

	#include <iostream>
	#include <sys/types.h>
	#include <sys/socket.h>
	#include <netinet/in.h>
	#include <arpa/inet.h>
	#include <unistd.h>
	#include <cstring>
	#include <signal.h>
	
	using namespace std;
	
	// 子进程处理函数
	void child_process(int client_sockfd) {
	  // 接收客户端发送的数据
	  char recv_buffer[1024] = {0};
	  int recv_result = recv(client_sockfd, recv_buffer, 1024, 0);
	  if (recv_result == -1) {
	    perror("recv");
	    return;
	  }
	
	  // 输出客户端发送的数据
	  cout << "Received from client: " << recv_buffer << endl;
	
	  // 向客户端发送数据
	  char send_buffer[] = "Hello from server!";
	  int send_result = send(client_sockfd, send_buffer, strlen(send_buffer), 0);
	  if (send_result == -1) {
	    perror("send");
	    return;
	  }
	
	  // 关闭socket
	  close(client_sockfd);
	
	  // 退出子进程
	  exit(0);
	}
	
	int main() {
	  // 创建socket
	  int server_sockfd = socket(AF_INET, SOCK_STREAM, 0);
	  if (server_sockfd == -1) {
	    perror("socket");
	    return -1;
	  }
	
	  // 绑定socket到IP和端口
	  struct sockaddr_in server_addr;
	  memset(&server_addr, 0, sizeof(server_addr));
	  server_addr.sin_family = AF_INET;
	  server_addr.sin_port = htons(8080); // 端口号
	  server_addr.sin_addr.s_addr = htonl(INADDR_ANY); // 绑定的IP地址是本机所有IP地址
	
	  int bind_result = bind(server_sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr));
	  if (bind_result == -1) {
	    perror("bind");
	    return -1;
	  }
	
	  // 监听socket
	  int listen_result = listen(server_sockfd, 5); // 最大同时监听5个连接
	  if (listen_result == -1) {
	    perror("listen");
	    return -1;
	  }
	
	  // 忽略SIGCHLD信号，防止僵尸进程产生
	  signal(SIGCHLD, SIG_IGN);
	
	  while (true) {
	    // 接受连接
	    struct sockaddr_in client_addr; // 客户端地址信息
	    socklen_t client_addr_len = sizeof(client_addr);
	
	    int client_sockfd = accept(server_sockfd, (struct sockaddr *)&client_addr, &client_addr_len);
	    if (client_sockfd == -1) {
	      perror("accept");
	      continue;
	    }
	
	    // 创建子进程处理客户端连接
	    pid_t pid = fork();
	    if (pid == -1) {
	      perror("fork");
	      close(client_sockfd);
	      continue;
	    } else if (pid == 0) {
	      // 子进程
	      close(server_sockfd); // 子进程不需要监听socket
	      child_process(client_sockfd);
	    } else {
	      // 父进程
	      close(client_sockfd); // 父进程不需要处理客户端连接
	    }
	  }
	
	  // 关闭socket
	  close(server_sockfd);
	
	  return 0;
	}
<br>
用select的方式， 单线程条件下， 实现多客户端的支持。
	
	#include <iostream>
	#include <sys/types.h>
	#include <sys/socket.h>
	#include <netinet/in.h>
	#include <arpa/inet.h>
	#include <unistd.h>
	#include <cstring>
	#include <sys/select.h>
	
	using namespace std;
	
	int main() {
	  // 创建socket
	  int server_sockfd = socket(AF_INET, SOCK_STREAM, 0);
	  if (server_sockfd == -1) {
	    perror("socket");
	    return -1;
	  }
	
	  // 绑定socket到IP和端口
	  struct sockaddr_in server_addr;
	  memset(&server_addr, 0, sizeof(server_addr));
	  server_addr.sin_family = AF_INET;
	  server_addr.sin_port = htons(8888); // 端口号
	  server_addr.sin_addr.s_addr = htonl(INADDR_ANY); // 绑定的IP地址是本机所有IP地址
	
	  int bind_result = bind(server_sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr));
	  if (bind_result == -1) {
	    perror("bind");
	    return -1;
	  }
	
	  // 监听socket
	  int listen_result = listen(server_sockfd, 5); // 最大同时监听5个连接
	  if (listen_result == -1) {
	    perror("listen");
	    return -1;
	  }
	
	  // 创建select读写集合
	  fd_set read_fds;
	  fd_set write_fds;
	
	  // 将server_sockfd添加到select读写集合中
	  FD_ZERO(&read_fds);
	  FD_ZERO(&write_fds);
	  FD_SET(server_sockfd, &read_fds);
	
	  while (true) {
	    // 使用select监听socket事件
	    int select_result = select(server_sockfd + 1, &read_fds, &write_fds, NULL, NULL);
	    if (select_result == -1) {
	      perror("select");
	      continue;
	    }
	
	    // 处理server_sockfd的新连接请求
	    if (FD_ISSET(server_sockfd, &read_fds)) {
	      struct sockaddr_in client_addr; // 客户端地址信息
	      socklen_t client_addr_len = sizeof(client_addr);
	
	      int client_sockfd = accept(server_sockfd, (struct sockaddr *)&client_addr, &client_addr_len);
	      if (client_sockfd == -1) {
	        perror("accept");
	        continue;
	      }
	
	      // 将client_sockfd添加到select读写集合中
	      FD_SET(client_sockfd, &read_fds);
	    }
	
	    // 处理客户端发送的数据
	    for (int i = 0; i <= server_sockfd; i++) {
	      if (FD_ISSET(i, &read_fds) && i != server_sockfd) {
	        // 接收客户端发送的数据
	        char recv_buffer[1024] = {0};
	        int recv_result = recv(i, recv_buffer, 1024, 0);
	        if (recv_result == -1) {
	          perror("recv");
	          FD_CLR(i, &read_fds); // 从select读写集合中删除client_sockfd
	          close(i);
	          continue;
	        } else if (recv_result == 0) {
	          // 客户端断开连接
	          FD_CLR(i, &read_fds); // 从select读写集合中删除client_sockfd
	          close(i);
	          continue;
	        }
	
	        // 输出客户端发送的数据
	        cout << "Received from client: " << recv_buffer << endl;
	
	        // 将client_sockfd添加到select写集合中，准备发送数据
	        FD_SET(i, &write_fds);
	      }
	    }
	
	    // 向客户端发送数据
	    for (int i = 0; i <= server_sockfd; i++) {
	      if (FD_ISSET(i, &write_fds) && i != server_sockfd) {
	        // 发送数据给客户端
	        char send_buffer[] = "Hello from server!";
	        int send_result = send(i, send_buffer, strlen(send_buffer), 0);
	        if (send_result == -1) {
	          perror("send");
	          FD_CLR(i, &write_fds); // 从select读写集合中删除client_sockfd
	          close(i);
	          continue;
	        }
	
	        // 将client_sockfd从select写集合中删除，等待下一次接收数据
	        FD_CLR(i, &write_fds);
	      }
	    }
	  }
	
	  // 关闭socket
	  close(server_sockfd);
	
	  return 0;
	}
<br>
c++语言实现一个简单的udp服务器和客户端。

服务器端代码：

	
	#include <iostream>
	#include <sys/types.h>
	#include <sys/socket.h>
	#include <netinet/in.h>
	#include <arpa/inet.h>
	#include <unistd.h>
	#include <cstring>
	
	using namespace std;
	
	int main() {
	  // 创建UDP socket
	  int server_sockfd = socket(AF_INET, SOCK_DGRAM, 0);
	  if (server_sockfd == -1) {
	    perror("socket");
	    return -1;
	  }
	
	  // 绑定socket到IP和端口
	  struct sockaddr_in server_addr;
	  memset(&server_addr, 0, sizeof(server_addr));
	  server_addr.sin_family = AF_INET;
	  server_addr.sin_port = htons(8888); // 端口号
	  server_addr.sin_addr.s_addr = htonl(INADDR_ANY); // 绑定的IP地址是本机所有IP地址
	
	  int bind_result = bind(server_sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr));
	  if (bind_result == -1) {
	    perror("bind");
	    return -1;
	  }
	
	  // 接收客户端发送的数据
	  char recv_buffer[1024] = {0};
	  struct sockaddr_in client_addr; // 客户端地址信息
	  socklen_t client_addr_len = sizeof(client_addr);
	
	  while (true) {
	    int recv_result = recvfrom(server_sockfd, recv_buffer, 1024, 0, (struct sockaddr *)&client_addr, &client_addr_len);
	    if (recv_result == -1) {
	      perror("recvfrom");
	      continue;
	    }
	
	    // 输出客户端发送的数据
	    cout << "Received from client: " << recv_buffer << endl;
	
	    // 向客户端发送数据
	    char send_buffer[] = "Hello from server!";
	    int send_result = sendto(server_sockfd, send_buffer, strlen(send_buffer), 0, (struct sockaddr *)&client_addr, client_addr_len);
	    if (send_result == -1) {
	      perror("sendto");
	      continue;
	    }
	  }
	
	  // 关闭socket
	  close(server_sockfd);
	
	  return 0;
	}
客户端代码：

	
	#include <iostream>
	#include <sys/types.h>
	#include <sys/socket.h>
	#include <netinet/in.h>
	#include <arpa/inet.h>
	#include <unistd.h>
	#include <cstring>
	
	using namespace std;
	
	int main() {
	  // 创建UDP socket
	  int client_sockfd = socket(AF_INET, SOCK_DGRAM, 0);
	  if (client_sockfd == -1) {
	    perror("socket");
	    return -1;
	  }
	
	  // 设置服务器地址和端口
	  struct sockaddr_in server_addr;
	  memset(&server_addr, 0, sizeof(server_addr));
	  server_addr.sin_family = AF_INET;
	  server_addr.sin_port = htons(8888); // 端口号
	  server_addr.sin_addr.s_addr = inet_addr("127.0.0.1"); // 服务器的IP地址
	
	  // 向服务器发送数据
	  char send_buffer[] = "Hello from client!";
	  int send_result = sendto(client_sockfd, send_buffer, strlen(send_buffer), 0, (struct sockaddr *)&server_addr, sizeof(server_addr));
	  if (send_result == -1) {
	    perror("sendto");
	    return -1;
	  }
	
	  // 接收服务器发送的数据
	  char recv_buffer[1024] = {0};
	  socklen_t server_addr_len = sizeof(server_addr);
	
	  int recv_result = recvfrom(client_sockfd, recv_buffer, 1024, 0, (struct sockaddr *)&server_addr, &server_addr_len);
	  if (recv_result == -1) {
	    perror("recvfrom");
	    return -1;
	  }
	
	  // 输出服务器发送的数据
	  cout << "Received from server: " << recv_buffer << endl;
	
	  // 关闭socket
	  close(client_sockfd);
	
	  return 0;
	}
<br>
C++使用单线程同时实现tcp和udp服务器。请给出代码
	
	#include <iostream>
	#include <sys/types.h>
	#include <sys/socket.h>
	#include <netinet/in.h>
	#include <arpa/inet.h>
	#include <unistd.h>
	#include <cstring>
	#include <sys/select.h>
	
	using namespace std;
	
	int main() {
	  // 创建TCP socket
	  int tcp_server_sockfd = socket(AF_INET, SOCK_STREAM, 0);
	  if (tcp_server_sockfd == -1) {
	    perror("socket");
	    return -1;
	  }
	
	  // 绑定TCP socket到IP和端口
	  struct sockaddr_in tcp_server_addr;
	  memset(&tcp_server_addr, 0, sizeof(tcp_server_addr));
	  tcp_server_addr.sin_family = AF_INET;
	  tcp_server_addr.sin_port = htons(8888); // 端口号
	  tcp_server_addr.sin_addr.s_addr = htonl(INADDR_ANY); // 绑定的IP地址是本机所有IP地址
	
	  int tcp_bind_result = bind(tcp_server_sockfd, (struct sockaddr *)&tcp_server_addr, sizeof(tcp_server_addr));
	  if (tcp_bind_result == -1) {
	    perror("bind");
	    return -1;
	  }
	
	  // 监听TCP socket
	  int tcp_listen_result = listen(tcp_server_sockfd, 5); // 最大同时监听5个连接
	  if (tcp_listen_result == -1) {
	    perror("listen");
	    return -1;
	  }
	
	  // 创建UDP socket
	  int udp_server_sockfd = socket(AF_INET, SOCK_DGRAM, 0);
	  if (udp_server_sockfd == -1) {
	    perror("socket");
	    return -1;
	  }
	
	  // 绑定UDP socket到IP和端口
	  struct sockaddr_in udp_server_addr;
	  memset(&udp_server_addr, 0, sizeof(udp_server_addr));
	  udp_server_addr.sin_family = AF_INET;
	  udp_server_addr.sin_port = htons(8888); // 端口号
	  udp_server_addr.sin_addr.s_addr = htonl(INADDR_ANY); // 绑定的IP地址是本机所有IP地址
	
	  int udp_bind_result = bind(udp_server_sockfd, (struct sockaddr *)&udp_server_addr, sizeof(udp_server_addr));
	  if (udp_bind_result == -1) {
	    perror("bind");
	    return -1;
	  }
	
	  // 创建select读写集合
	  fd_set read_fds;
	  fd_set write_fds;
	
	  // 将TCP socket和UDP socket添加到select读写集合中
	  FD_ZERO(&read_fds);
	  FD_ZERO(&write_fds);
	  FD_SET(tcp_server_sockfd, &read_fds);
	  FD_SET(udp_server_sockfd, &read_fds);
	
	  while (true) {
	    // 使用select监听socket事件
	    int select_result = select(max(tcp_server_sockfd, udp_server_sockfd) + 1, &read_fds, &write_fds, NULL, NULL);
	    if (select_result == -1) {
	      perror("select");
	      continue;
	    }
	
	    // 处理TCP socket的新连接请求
	    if (FD_ISSET(tcp_server_sockfd, &read_fds)) {
	      struct sockaddr_in client_addr; // 客户端地址信息
	      socklen_t client_addr_len = sizeof(client_addr);
	
	      int client_sockfd = accept(tcp_server_sockfd, (struct sockaddr *)&client_addr, &client_addr_len);
	      if (client_sockfd == -1) {
	        perror("accept");
	        continue;
	      }
	
	      // 将client_sockfd添加到select读写集合中
	      FD_SET(client_sockfd, &read_fds);
	    }
	
	    // 处理TCP客户端发送的数据
	    for (int i = 0; i <= tcp_server_sockfd; i++) {
	      if (FD_ISSET(i, &read_fds) && i != tcp_server_sockfd) {
	        // 接收TCP客户端发送的数据
	        char recv_buffer[1024] = {0};
	        int recv_result = recv(i, recv_buffer, 1024, 0);
	        if (recv_result == -1) {
	          perror("recv");
	          FD_CLR(i, &read_fds); // 从select读写集合中删除client_sockfd
	          close(i);
	          continue;
	        } else if (recv_result == 0) {
	          // TCP客户端断开连接
	          FD_CLR(i, &read_fds); // 从select读写集合中删除client_sockfd
	          close(i);
	          continue;
	        }
	
	        // 输出TCP客户端发送的数据
	        cout << "Received from TCP client: " << recv_buffer << endl;
	
	        // 将client_sockfd添加到select写集合中，准备发送数据
	        FD_SET(i, &write_fds);
	      }
	    }
	
	    // 处理UDP客户端发送的数据
	    if (FD_ISSET(udp_server_sockfd, &read_fds)) {
	      // 接收UDP客户端发送的数据
	      char recv_buffer[1024] = {0};
	      struct sockaddr_in client_addr; // 客户端地址信息
	      socklen_t client_addr_len = sizeof(client_addr);
	
	      int recv_result = recvfrom(udp_server_sockfd, recv_buffer, 1024, 0, (struct sockaddr *)&client_addr, &client_addr_len);
	      if (recv_result == -1) {
	        perror("recvfrom");
	        continue;
	      }
	
	      // 输出UDP客户端发送的数据
	      cout << "Received from UDP client: " << recv_buffer << endl;
	
	      // 向UDP客户端发送数据
	      char send_buffer[] = "Hello from server!";
	      int send_result = sendto(udp_server_sockfd, send_buffer, strlen(send_buffer), 0, (struct sockaddr *)&client_addr, client_addr_len);
	      if (send_result == -1) {
	        perror("sendto");
	        continue;
	      }
	    }
	
	    // 向TCP客户端发送数据
	    for (int i = 0; i <= tcp_server_sockfd; i++) {
	      if (FD_ISSET(i, &write_fds) && i != tcp_server_sockfd) {
	        // 发送数据给TCP客户端
	        char send_buffer[] = "Hello from server!";
	        int send_result = send(i, send_buffer, strlen(send_buffer), 0);
	        if (send_result == -1) {
	          perror("send");
	          FD_CLR(i, &write_fds); // 从select读写集合中删除client_sockfd
	          close(i);
	          continue;
	        }
	
	        // 将client_sockfd从select写集合中删除，等待下一次接收数据
	        FD_CLR(i, &write_fds);
	      }
	    }
	  }
	
	  // 关闭socket
	  close(tcp_server_sockfd);
	  close(udp_server_sockfd);
	
	  return 0;
	}