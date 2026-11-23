import { useEffect, useMemo, useState, useCallback } from "react";
import VideoStream, { Caption } from "./VideoStream";
import PersonList from "./PersonList";
import PersonDetail from "./PersonDetail";
import EventFeed from "./EventFeed";
import { api, EventMessage, PersonSummary } from "../services/api";
import { useWebSocket } from "../hooks/useWebSocket";
import KnownPeople, { knownPeopleChanged } from "./KnownPeople";
import FacesOnScreen from "./FacesOnScreen";

export default function Dashboard() {
  const [persons, setPersons] = useState<PersonSummary[]>([]);
  const [events, setEvents]   = useState<EventMessage[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [liveAnnotations, setLiveAnnotations] = useState<any[]>([]);
  const [caption, setCaption] = useState<Caption | null>(null);

  const refreshPersons = useCallback(async () => {
    try { setPersons(await api.listPersons()); } catch {}
  }, []);

  useEffect(() => {
    refreshPersons();
    // Reload the log after an enrolment too: naming someone from the video
    // also names what they did since they came into view.
    const load = () => api.recentEvents(200).then(setEvents).catch(() => {});
    load();
    knownPeopleChanged.addEventListener("changed", load);
    return () => knownPeopleChanged.removeEventListener("changed", load);
  }, [refreshPersons]);

  useWebSocket((msg) => {
    if (msg.type === "event") {
      setEvents((prev) => [msg, ...prev].slice(0, 200));
      refreshPersons();
    } else if (msg.type === "frame_annotations") {
      setLiveAnnotations(msg.annotations);
      if (msg.caption) setCaption(msg.caption);
    }
  });

  // Track ID -> name, from the newest event that named them.
  const names = useMemo(() => {
    const m: Record<string, string> = {};
    for (const e of events) if (e.identity && !(e.person_id in m)) m[e.person_id] = e.identity;
    return m;
  }, [events]);

  return (
    <div className="grid grid-cols-12 gap-6">
      <section className="col-span-12 xl:col-span-8 space-y-6">
        <VideoStream annotations={liveAnnotations} caption={caption} />
        <EventFeed events={events} onSelect={setSelected} />
      </section>

      <aside className="col-span-12 xl:col-span-4 space-y-6">
        <div id="known-people" className="scroll-mt-24 space-y-6"><FacesOnScreen /><KnownPeople /></div>
        <PersonList persons={persons} names={names} selected={selected} onSelect={setSelected} />
        {selected && <PersonDetail personId={selected} />}
      </aside>
    </div>
  );
}
