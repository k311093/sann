-module(cirno).

-behaviour(gen_server).

-export([start_link/0, run/1, call/2, stop/0]).
-export([init/1, handle_call/3, handle_cast/2, handle_info/2, code_change/3, terminate/2]).

cirno_node_id() ->
    list_to_atom(
        lists:flatten(
            string:sub_word(atom_to_list(node()), 1, $@),
            "_cirno")
    ).

start_link() ->
    Id = cirno_node_id(),
	gen_server:start_link({local, Id}, ?MODULE, [Id], []).

run(Code) when is_list(Code) ->
	gen_server:call(cirno_node_id(), {exec, list_to_binary(Code)}, infinity);
run(Code) when is_binary(Code) ->
	gen_server:call(cirno_node_id(), {exec, Code}, infinity).

call(Fun, Args) when is_atom(Fun), is_list(Args) ->
	gen_server:call(cirno_node_id(), {call, Fun, Args}, infinity).

stop() ->
	gen_server:call(cirno_node_id(), stop, infinity).


-record(state, {
	id,
	port,
	mbox,
	from,
	infotext = [],
	infoline = []
}).

-define(MAX_INFOTEXT_LINES, 1000).

init([Id]) ->
	process_flag(trap_exit, true),
	{Clean_Id, Host, Cirno_Node_Name} = mk_node_name(Id),
	{Result, Cmd_or_Error} = case os:find_executable("cirno_enode", ".") of
		false ->
			{stop, cirno_not_found};
		Cirno ->
			{ok, mk_cmdline(Cirno, Clean_Id, Host, 0)}
	end,
	case {Result, Cmd_or_Error} of
		{stop, Error} ->
			{stop, Error};
		{ok, Cmd} ->
			Port = open_port({spawn, Cmd}, [stream, {line, 100}, stderr_to_stdout, exit_status]),
			wait_for_startup(#state{id=Id, port=Port, mbox={cirno, Cirno_Node_Name}})
	end.

mk_cmdline(Cirno, Id, Host, Tracelevel) ->
	lists:flatten([
		Cirno,
		quote(Id),
		quote(Host),
		quote(atom_to_list(node())),
		quote(atom_to_list(erlang:get_cookie())),
		quote(integer_to_list(Tracelevel))
	]).

wait_for_startup(#state{port=Port} = State) ->
	receive
		{Port, {exit_status, N}} ->
			{stop, {exit_status, N}};
		{Port, {data, {eol, "READY"}}} ->
			{ok, State};
		{Port, {data, {eol, "."}}} ->
			wait_for_startup(State);
		{Port, {data, {eol, _S}}} ->
			wait_for_startup(State)
	end.


handle_call({exec, Code}, From, #state{mbox=Mbox, from=undefined} = State) ->
	Mbox ! {exec, self(), Code, []},
	{noreply, State#state{from=From}};
handle_call({call, Fun, Args}, From, #state{mbox=Mbox, from=undefined} = State) ->
	Mbox ! {call, self(), Fun, Args},
	{noreply, State#state{from=From}};
handle_call(stop, _From, #state{from=undefined} = State) ->
	{stop, normal, ok, State};
handle_call(_Request, _From, #state{from=Id} = State) when Id =/= undefined ->
	{reply, {error, busy}, State}.


handle_cast(_Request, State) ->
	{noreply, State}.


handle_info({Port, {exit_status, 0}}, #state{port=Port} = State) ->
	{stop, normal, State#state{port=undefined, mbox=undefined}};
handle_info({Port, {exit_status, N}}, #state{port=Port} = State) ->
	{stop, {port_status, N}, State#state{port=undefined, mbox=undefined}};
handle_info({'EXIT', Port, Reason}, #state{port=Port} = State) ->
	{stop, {port_exit, Reason}, State#state{port=undefined, mbox=undefined}};


handle_info({Port, {data, {noeol, S}}}, #state{port=Port} = State) ->
	{noreply, noeol_port_data(S, State)};
handle_info({Port, {data, {eol, "."}}}, #state{port=Port, infoline = []} = State) ->
	{noreply, flush_port_data(State)};
handle_info({Port, {data, {eol, S}}}, #state{port=Port} = State) ->
	{noreply, eol_port_data(S, State)};


handle_info({error, _Reason} = Error, #state{from=From} = State) when From =/= undefined ->
	gen_server:reply(From, Error),
	{noreply, State#state{from=undefined}};
handle_info({cirno, _Result} = Reply, #state{from=From} = State) when From =/= undefined ->
	gen_server:reply(From, Reply),
	{noreply, State#state{from=undefined}};


handle_info(_Info, State) ->
	{noreply, State}.


terminate(_Reason, #state{mbox=undefined} = _State) ->
	ok;

terminate(_Reason, #state{mbox=Mbox} = State) ->
	Mbox ! {stop, self(), [], []},
	wait_for_exit(State).

wait_for_exit(#state{port=Port} = State) ->
	receive
		{Port, {exit_status, 0}} ->
			ok;
		{Port, {exit_status, _N}} ->
			ok;
		{'EXIT', _Port, _Reason} ->
			ok;
		{Port, {data, {eol, "."}}} ->
			wait_for_exit(flush_port_data(State));
		{Port, {data, {noeol, S}}} ->
			wait_for_exit(noeol_port_data(S, State));
		{Port, {data, {eol, S}}} ->
			wait_for_exit(eol_port_data(S, State));
		_ ->
			wait_for_exit(State)
	end.

code_change(_Old, State, _Extra) ->
	{ok, State}.


noeol_port_data(S, #state{infotext = Text, infoline = []} = State)
		when length(Text) >= ?MAX_INFOTEXT_LINES ->
	noeol_port_data(S, flush_port_data(State));
noeol_port_data(S, #state{infoline = Line} = State) ->
	State#state{infoline = [S | Line]}.


eol_port_data(S, #state{infotext = Text, infoline = []} = State)
		when length(Text) >= ?MAX_INFOTEXT_LINES ->
	eol_port_data(S, flush_port_data(State));
eol_port_data(S, #state{infotext = Text, infoline = Line} = State) ->
	Full_Line = lists:flatten(lists:reverse([S | Line])),
	State#state{infotext = [Full_Line | Text], infoline = []}.


flush_port_data(#state{infotext = [], infoline = []} = State) ->
	State;
flush_port_data(#state{infoline = [_ | _]} = State) ->
	flush_port_data(eol_port_data("", State));
flush_port_data(#state{infotext = _Text} = State) ->
	State#state{infotext = [], infoline = []}.


mk_node_name(Id) ->
	This_Id = re:replace(atom_to_list(Id), "[^_0-9a-zA-Z]+", "_", [global, {return, list}]),
	This_Host = string:sub_word(atom_to_list(node()), 2, $@),
	{This_Id, This_Host, list_to_atom(lists:flatten([This_Id, "@", This_Host]))}.

quote(S) ->
	case ostype() of
		win32 -> [" \"", S, "\""];
		unix -> [" '", S, "'"]
	end.

ostype() ->
	case os:type() of
		{Type, _} -> Type;
		Type -> Type
	end.

